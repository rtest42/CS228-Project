package fingerprintmodel;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;

public class CustomFingerprintPairwise {
    /**
     * Evaluates a custom CNN fingerprint encoder on the same pairs as the
     * SourceAFIS baseline.
     *
     * <p>Each unique image is loaded and encoded once. The resulting embeddings
     * are cached and reused for all pairwise distance calculations.
     *
     * <p>Pair comparisons are distributed across CPU threads. Completed chunks
     * are written to a CSV file by the main thread, which also aggregates the
     * confusion matrix.
     *
     * <p>The match threshold is applied to Euclidean embedding distances:
     * distances less than or equal to the threshold predict a genuine match.
     *
     * @throws Exception if image loading, model inference, pair parsing,
     *                   comparison, or output writing fails
     */
    public static void run() throws Exception {
        System.setProperty(
                "jna.library.path",
                System.getProperty("user.home")
                        + "/.djl.ai/pytorch/2.7.1-cu128-win-x86_64"
        );

        final String enhanced = (Boolean.getBoolean("enhanced") ? "_enhanced" : "");
        final Path modelPath = Path.of(
                "ml_pipeline/checkpoints/fingerprint_encoder" + enhanced + ".pt"
        );
        final Path pairsPath = Path.of(
                "ml_pipeline/data/pairwise_scores_baseline.csv"
        );
        final Path outputPath = Path.of(
                "ml_pipeline/data/custom_encoder_scores" + enhanced + ".csv"
        );

        final int threads = Runtime.getRuntime().availableProcessors();
        final int chunkSize = 256;
        final int maxInFlight = Math.max(1, threads * 2);

        long start = System.nanoTime();

        if (!Files.isRegularFile(modelPath)) {
            throw new IOException("Model not found: " + modelPath);
        }
        if (!Files.isRegularFile(pairsPath)) {
            throw new IOException("Pairwise CSV not found: " + pairsPath);
        }

        IO.println("Available processors: " + threads);

        // First pass: collect unique image paths from the baseline pairs.
        Set<String> uniquePaths = loadUniquePaths(pairsPath);
        IO.println("Unique images: " + uniquePaths.size());

        // Load and encode each image exactly once.
        Map<String, float[]> embeddings = new HashMap<>();

        final float threshold = Float.parseFloat(System.getProperty("custom_threshold", "0.40"));

        try (FingerprintEncoderService verifier =
                     new FingerprintEncoderService(modelPath, threshold)) {

            try (ProgressBar bar = new ProgressBarBuilder()
                    .setTaskName("Encoding images")
                    .setInitialMax(uniquePaths.size())
                    .setStyle(ProgressBarStyle.ASCII)
                    .continuousUpdate()
                    .build()) {

                for (String imagePath : uniquePaths) {
                    float[] pixels = FingerprintImageLoader.load(imagePath);
                    float[] embedding = verifier.embed(pixels);

                    embeddings.put(imagePath, embedding);
                    bar.step();
                }
            }

            IO.println("Embedding extraction complete.");

            // The embedding cache is no longer modified after this point.
            // Workers only read the cached arrays and calculate distances.
            ExecutorService pool = Executors.newFixedThreadPool(threads);

            try (BufferedReader reader = Files.newBufferedReader(
                    pairsPath, StandardCharsets.UTF_8);
                 BufferedWriter writer = Files.newBufferedWriter(
                         outputPath, StandardCharsets.UTF_8)) {

                String header = reader.readLine();
                if (header == null) {
                    throw new IOException("Pairwise CSV is empty.");
                }

                writer.write(
                        "image_path_1,image_path_2,label,distance,prediction"
                );
                writer.newLine();



                long expectedPairs = countPairs(pairsPath);

                try (ProgressBar bar = new ProgressBarBuilder()
                        .setTaskName("Comparing pairs")
                        .setInitialMax(expectedPairs)
                        .setStyle(ProgressBarStyle.ASCII)
                        .continuousUpdate()
                        .build()) {

                    ExecutorCompletionService<ChunkResult> completion =
                            new ExecutorCompletionService<>(pool);

                    List<Pair> chunk = new ArrayList<>(chunkSize);

                    long tp = 0;
                    long tn = 0;
                    long fp = 0;
                    long fn = 0;
                    long totalPairs = 0;
                    long completedPairs = 0;
                    int inFlight = 0;

                    String line;

                    while ((line = reader.readLine()) != null) {
                        if (line.isBlank()) {
                            continue;
                        }

                        chunk.add(parsePair(line));
                        totalPairs++;

                        if (chunk.size() == chunkSize) {
                            List<Pair> batch = List.copyOf(chunk);
                            chunk.clear();

                            completion.submit(
                                    compareChunk(batch, embeddings, verifier)
                            );
                            inFlight++;

                            // Apply backpressure so queued work stays bounded.
                            if (inFlight >= maxInFlight) {
                                ChunkResult result = getResult(completion);
                                inFlight--;

                                writeRows(writer, result.rows());

                                tp += result.tp();
                                tn += result.tn();
                                fp += result.fp();
                                fn += result.fn();

                                int processed = result.rows().size();
                                completedPairs += processed;
                                bar.stepBy(processed);
                            }
                        }
                    }

                    // Submit the final partial chunk.
                    if (!chunk.isEmpty()) {
                        completion.submit(
                                compareChunk(
                                        List.copyOf(chunk), embeddings, verifier
                                )
                        );
                        inFlight++;
                    }

                    // Consume every outstanding task.
                    while (inFlight > 0) {
                        ChunkResult result = getResult(completion);
                        inFlight--;

                        writeRows(writer, result.rows());

                        tp += result.tp();
                        tn += result.tn();
                        fp += result.fp();
                        fn += result.fn();

                        int processed = result.rows().size();
                        completedPairs += processed;
                        bar.stepBy(processed);
                    }

                    writer.flush();

                    if (completedPairs != totalPairs) {
                        throw new IllegalStateException(
                                "Pair count mismatch: read " + totalPairs
                                        + ", completed " + completedPairs
                        );
                    }

                    printStats(tp, tn, fp, fn);
                }

            } finally {
                pool.shutdown();

                if (!pool.awaitTermination(1, TimeUnit.MINUTES)) {
                    pool.shutdownNow();

                    if (!pool.awaitTermination(1, TimeUnit.MINUTES)) {
                        IO.println("Warning: worker pool did not terminate.");
                    }
                }
            }
        }

        IO.println("\nTotal pairs evaluated: "
                + (Files.lines(pairsPath).count() - 1));
        IO.println("Output: " + outputPath.toAbsolutePath());

        System.out.printf(
                "Elapsed: %.2f seconds%n",
                (System.nanoTime() - start) / 1e9
        );
    }

    /**
     * Represents one input pair from the baseline CSV.
     *
     * @param path1 first image path
     * @param path2 second image path
     * @param label 1 for the same finger, 0 otherwise
     */
    record Pair(String path1, String path2, int label) {}

    /**
     * Stores the results of one completed comparison chunk.
     *
     * @param rows formatted output CSV rows
     * @param tp true positives
     * @param tn true negatives
     * @param fp false positives
     * @param fn false negatives
     */
    record ChunkResult(
            List<String> rows,
            long tp,
            long tn,
            long fp,
            long fn) {}

    /**
     * Loads the unique image paths referenced by the baseline CSV.
     *
     * @param pairsPath input pairwise CSV
     * @return distinct image paths in encounter order
     * @throws IOException if the CSV cannot be read
     */
    private static Set<String> loadUniquePaths(Path pairsPath)
            throws IOException {
        Set<String> paths = new LinkedHashSet<>();

        try (BufferedReader reader = Files.newBufferedReader(
                pairsPath, StandardCharsets.UTF_8)) {

            if (reader.readLine() == null) {
                throw new IOException("Pairwise CSV is empty.");
            }

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }

                Pair pair = parsePair(line);
                paths.add(pair.path1());
                paths.add(pair.path2());
            }
        }

        return paths;
    }

    /**
     * Parses a baseline CSV row.
     *
     * <p>This parser assumes image paths do not contain commas. Use a proper
     * CSV library if commas can occur inside quoted fields.
     *
     * @param line input CSV row
     * @return parsed pair and its ground-truth label
     */
    private static Pair parsePair(String line) {
        String[] fields = line.split(",", -1);

        if (fields.length < 8) {
            throw new IllegalArgumentException(
                    "Malformed baseline CSV row: " + line
            );
        }

        String path1 = unquote(fields[0]);
        String path2 = unquote(fields[1]);
        int label = Integer.parseInt(fields[6].trim());

        if (label != 0 && label != 1) {
            throw new IllegalArgumentException(
                    "Invalid ground-truth label: " + label
            );
        }

        return new Pair(path1, path2, label);
    }

    /**
     * Creates a task that compares cached embeddings for a chunk of pairs.
     *
     * <p>The task does not perform model inference. The embedding-distance
     * and threshold methods operate on Java arrays and do not access the
     * predictor or NDManager.
     *
     * @param pairs pairs to compare
     * @param embeddings cached embeddings keyed by image path
     * @param verifier custom fingerprint encoder service
     * @return a task containing result rows and local confusion counts
     */
    private static Callable<ChunkResult> compareChunk(
            List<Pair> pairs,
            Map<String, float[]> embeddings,
            FingerprintEncoderService verifier) {

        return () -> {
            List<String> rows = new ArrayList<>(pairs.size());

            long tp = 0;
            long tn = 0;
            long fp = 0;
            long fn = 0;

            for (Pair pair : pairs) {
                float[] embedding1 = embeddings.get(pair.path1());
                float[] embedding2 = embeddings.get(pair.path2());

                if (embedding1 == null || embedding2 == null) {
                    throw new IllegalStateException(
                            "Missing cached embedding for "
                                    + pair.path1() + " or " + pair.path2()
                    );
                }

                float distance = verifier.embeddingDistance(
                        embedding1, embedding2
                );

                int prediction =
                        verifier.isGenuineMatch(distance) ? 1 : 0;

                if (pair.label() == 1 && prediction == 1) {
                    tp++;
                } else if (pair.label() == 0 && prediction == 0) {
                    tn++;
                } else if (pair.label() == 0) {
                    fp++;
                } else {
                    fn++;
                }

                rows.add(
                        csv(pair.path1()) + ","
                                + csv(pair.path2()) + ","
                                + pair.label() + ","
                                + distance + ","
                                + prediction
                );
            }

            return new ChunkResult(rows, tp, tn, fp, fn);
        };
    }

    /**
     * Retrieves a completed task and propagates task failures.
     *
     * @param completion completion service
     * @return completed chunk result
     * @throws InterruptedException if interrupted while waiting
     * @throws ExecutionException if a comparison task fails
     */
    private static ChunkResult getResult(
            ExecutorCompletionService<ChunkResult> completion)
            throws InterruptedException, ExecutionException {
        return completion.take().get();
    }

    /**
     * Writes the result rows for a completed chunk.
     *
     * @param writer output CSV writer
     * @param rows formatted CSV rows
     * @throws IOException if writing fails
     */
    private static void writeRows(
            BufferedWriter writer,
            List<String> rows) throws IOException {

        for (String row : rows) {
            writer.write(row);
            writer.newLine();
        }
    }

    /**
     * Prints the aggregate confusion matrix and evaluation metrics.
     *
     * @param tp true positives
     * @param tn true negatives
     * @param fp false positives
     * @param fn false negatives
     */
    private static void printStats(
            long tp, long tn, long fp, long fn) {

        long total = tp + tn + fp + fn;

        double accuracy = divide(tp + tn, total);
        double precision = divide(tp, tp + fp);
        double recall = divide(tp, tp + fn);
        double specificity = divide(tn, tn + fp);
        double f1 = divide(2 * tp, 2 * tp + fp + fn);
        double fpr = divide(fp, fp + tn);
        double fnr = divide(fn, fn + tp);

        IO.println("\nConfusion Matrix");
        IO.println("----------------");
        System.out.printf("TP: %,d | FN: %,d%n", tp, fn);
        System.out.printf("FP: %,d | TN: %,d%n", fp, tn);

        IO.println("\nEvaluation Metrics");
        IO.println("-----------------");
        System.out.printf("Accuracy:    %.4f%n", accuracy);
        System.out.printf("Precision:   %.4f%n", precision);
        System.out.printf("Recall / TPR: %.4f%n", recall);
        System.out.printf("Specificity: %.4f%n", specificity);
        System.out.printf("F1-score:    %.4f%n", f1);
        System.out.printf("FPR:         %.4f%n", fpr);
        System.out.printf("FNR:         %.4f%n", fnr);
        System.out.printf("Total pairs: %,d%n", total);
    }

    /**
     * Removes optional surrounding CSV quotes and trims whitespace.
     *
     * @param value raw CSV field
     * @return unquoted field
     */
    private static String unquote(String value) {
        value = value.trim();

        if (value.length() >= 2
                && value.startsWith("\"")
                && value.endsWith("\"")) {
            return value.substring(1, value.length() - 1)
                    .replace("\"\"", "\"");
        }

        return value;
    }

    /**
     * Quotes a CSV field and escapes embedded double quotes.
     *
     * @param value field value
     * @return escaped CSV field
     */
    private static String csv(String value) {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }

    /**
     * Divides two values, returning zero if the denominator is zero.
     *
     * @param numerator numerator
     * @param denominator denominator
     * @return quotient or zero for a zero denominator
     */
    private static double divide(long numerator, long denominator) {
        return denominator == 0
                ? 0.0
                : (double) numerator / denominator;
    }

    private static long countPairs(Path pairsPath) throws IOException {
        try (var lines = Files.lines(pairsPath)) {
            return Math.max(0, lines.count() - 1);
        }
    }

    static void main(String[] args) throws Exception {
        run();
    }
}