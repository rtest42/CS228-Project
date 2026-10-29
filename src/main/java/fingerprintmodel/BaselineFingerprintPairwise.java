package fingerprintmodel;

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.jetbrains.annotations.NotNull;

public class BaselineFingerprintPairwise {
    private static final int THREADS = Runtime.getRuntime().availableProcessors();
    private static final String PREPEND_DIR = "ml_pipeline/";
    private static final String ENHANCED = (System.getProperty("enhanced") != null ? "_enhanced" : "");
    private static final Path MANIFEST = Paths.get(PREPEND_DIR + "data/manifest" + ENHANCED + ".csv");
    private static final Path OUTPUT = Paths.get(PREPEND_DIR + "data/pairwise_scores_baseline.csv");
    private static final double MATCH_THRESHOLD = 40.0;

    private static final LongAdder tp = new LongAdder();
    private static final LongAdder tn = new LongAdder();
    private static final LongAdder fp = new LongAdder();
    private static final LongAdder fn = new LongAdder();

    record Fingerprint(
            String imagePath,
            String classId,
            String impressionId,
            FingerprintTemplate template) {
    }

    /**
     * Generates baseline pairwise fingerprint matching scores for all unique pairs
     * of fingerprints listed in the manifest.
     *
     * <p>The method performs the following steps:
     * <ol>
     *     <li>Loads fingerprint metadata from the manifest CSV file.</li>
     *     <li>Extracts fingerprint templates using SourceAFIS.</li>
     *     <li>Compares every unordered pair of templates.</li>
     *     <li>Assigns a label of {@code 1} if both fingerprints share the same
     *     finger ID, or {@code 0} otherwise.</li>
     *     <li>Writes the image paths, fingerprint identifiers, impression
     *     identifiers, labels, and matching scores to the output CSV file.</li>
     * </ol>
     *
     * @throws Exception if manifest loading, fingerprint image processing,
     * pairwise comparison, output writing, or task execution fails
     */
    public static void run() throws Exception {
        long start = System.nanoTime();

        IO.println("Available processors: " + THREADS);

        List<Fingerprint> fingerprints = loadManifest(MANIFEST);
        IO.println("Manifest entries: " + fingerprints.size());

        ExecutorService pool = Executors.newFixedThreadPool(THREADS);

        try {
            // Feature extraction, parallelized across CPUs.
            List<Future<Fingerprint>> futures = new ArrayList<>();

            for (Fingerprint f : fingerprints) {
                futures.add(pool.submit(() -> {
                    Path path = Paths.get(PREPEND_DIR + f.imagePath());

                    FingerprintTemplate template = new FingerprintTemplate(
                            new FingerprintImage(Files.readAllBytes(path)));

                    return new Fingerprint(
                            PREPEND_DIR + f.imagePath(),
                            f.classId(),
                            f.impressionId(),
                            template);
                }));
            }

            List<Fingerprint> ready = new ArrayList<>();
            for (Future<Fingerprint> future : futures) {
                ready.add(future.get());
            }

            IO.println("Templates extracted: " + ready.size());

            long totalPairs = (long) ready.size() * (ready.size() + 1) / 2;

            IO.println("Total pairs: " + totalPairs);

            // Open output CSV.
            try (BufferedWriter writer = Files.newBufferedWriter(
                    OUTPUT, StandardCharsets.UTF_8)) {

                writer.write("image_path_1,image_path_2," +
                        "finger_id_1,finger_id_2," +
                        "impression_id_1,impression_id_2," +
                        "label,score");
                writer.newLine();

                // Chunk work to avoid creating tons of individual tasks.
                List<Callable<List<String>>> tasks = getTasks(ready);

                long completed = 0;

                // ExecutorCompletionService allows results to be written
                // as chunks finish, instead of holding all rows in RAM.
                ExecutorCompletionService<List<String>> completion =
                        new ExecutorCompletionService<>(pool);

                for (Callable<List<String>> task : tasks) {
                    completion.submit(task);
                }

                // Retrieve results.
                for (int i = 0; i < tasks.size(); i++) {
                    List<String> rows = completion.take().get();

                    for (String row : rows) {
                        writer.write(row);
                        writer.newLine();
                    }

                    completed += rows.size();
                    if (completed % 23064 < rows.size() || completed == totalPairs) {
                        System.out.printf("Compared %,d / %,d pairs%n", completed, totalPairs);
                    }
                }
            }

            IO.println("Output: " + OUTPUT.toAbsolutePath());

            printStats();
        } finally {
            // Shut down the thread pool.
            pool.shutdown();
            if (!pool.awaitTermination(1, TimeUnit.MINUTES)) {
                pool.shutdownNow();
            }
        }

        System.out.printf("Elapsed: %.2f seconds%n", (System.nanoTime() - start) / 1e9);
    }

    /**
     * Helper function for printing out baseline metrics.
     */
    private static void printStats() {
        long TP = tp.sum();
        long TN = tn.sum();
        long FP = fp.sum();
        long FN = fn.sum();

        double accuracy = divide(TP + TN, TP + TN + FP + FN);
        double precision = divide(TP, TP + FP);
        double recall = divide(TP, TP + FN);
        double tnr = divide(TN, TN + FP);
        double f1 = divide(2 * TP, 2 * TP + FP + FN);
        double fpr = divide(FP, FP + TN);
        double fnr = divide(FN, FN + TP);

        IO.println("\nConfusion Matrix");
        IO.println("----------------");
        IO.println("TP: " + TP + " | FN: " + FN);
        IO.println("FP: " + FP + " | TN: " + TN);

        IO.println("\nEvaluation Metrics");
        IO.println("-----------------");
        System.out.printf("Accuracy:    %.4f%n", accuracy);
        System.out.printf("Precision:   %.4f%n", precision);
        System.out.printf("Recall/TPR:  %.4f%n", recall);
        System.out.printf("TNR:         %.4f%n", tnr);
        System.out.printf("F1-score:    %.4f%n", f1);
        System.out.printf("FPR:         %.4f%n", fpr);
        System.out.printf("FNR:         %.4f%n", fnr);
    }

    /**
     * Helper method for division with division by zero handling.
     *
     * @param numerator the top number in a fraction
     * @param denominator the bottom number in a fraction
     * @return the numerator divided by the denominator or zero if the denominator is zero
     */
    private static double divide(long numerator, long denominator) {
        return denominator == 0 ? 0.0 : (double) numerator / denominator;
    }

    /**
     * Divides comparison work into chunks rather than creating
     * hundreds of thousands of individual tasks.
     *
     * @param ready the list of Fingerprint objects
     * @return the list of Fingerprint objects split into chunks of size 256
     */
    @NotNull
    private static List<Callable<List<String>>> getTasks(List<Fingerprint> ready) {
        final int chunkSize = 256;
        List<Callable<List<String>>> tasks = new ArrayList<>();

        for (int i = 0; i < ready.size(); i += chunkSize) {
            final int from = i;
            final int to = Math.min(i + chunkSize, ready.size());

            tasks.add(() -> {
                List<String> rows = new ArrayList<>();

                for (int i1 = from; i1 < to; i1++) {
                    Fingerprint a = ready.get(i1);

                    // One matcher per probe, reused for candidates.
                    FingerprintMatcher matcher = new FingerprintMatcher(a.template());

                    // Compute each pair and assign its label.
                    // Also include two of the same impressions as a pair.
                    for (int j = i1; j < ready.size(); j++) {
                        Fingerprint b = ready.get(j);

                        double score = matcher.match(b.template());

                        int label = a.classId().equals(b.classId()) ? 1 : 0;

                        int prediction = score >= MATCH_THRESHOLD ? 1 : 0;

                        if (label == 1 && prediction == 1) {
                            tp.increment();
                        } else if (label == 0 && prediction == 0) {
                            tn.increment();
                        } else if (label == 0) { // prediction = 1
                            fp.increment();
                        } else { // label = 1, prediction = 0
                            fn.increment();
                        }

                        rows.add(csv(a.imagePath()) + "," +
                                csv(b.imagePath()) + "," +
                                csv(a.classId()) + "," +
                                csv(b.classId()) + "," +
                                csv(a.impressionId()) + "," +
                                csv(b.impressionId()) + "," +
                                label + "," +
                                score);
                    }
                }

                return rows;
            });
        }

        return tasks;
    }

    /**
     * Loads fingerprint metadata from a CSV manifest and validates that
     * each referenced image file exists.
     *
     * <p>Each valid data row is converted into a {@link Fingerprint} object.
     * The fingerprint template is initially {@code null} because image loading
     * and template extraction are performed separately.
     *
     * @param manifest path to the CSV manifest containing fingerprint metadata
     * @return a list of fingerprint records parsed from the manifest
     * @throws IOException if the manifest is empty, required columns are missing,
     * a row is malformed, or a referenced image does not exist
     */
    private static List<Fingerprint> loadManifest(Path manifest) throws IOException {
        List<Fingerprint> result = new ArrayList<>();

        // Open CSV file.
        try (BufferedReader reader = Files.newBufferedReader(manifest, StandardCharsets.UTF_8)) {
            // Read and validate header.
            String header = reader.readLine();
            if (header == null) {
                throw new IOException("Manifest is empty");
            }

            // Map column names to indices.
            Map<String, Integer> columns = Arrays.stream(
                    header.replace("\uFEFF", "").split(",", -1))
                    .map(String::trim)
                    .collect(Collectors.toMap(
                            s -> s,
                            new Function<>() {
                                private int index = 0;

                                @Override
                                public Integer apply(String ignored) {
                                    return index++;
                                }
                            },
                            (a, b) -> a,
                            LinkedHashMap::new));

            // Verifies the columns exist
            final List<String> columnRequirements = List.of("image_path", "class_id", "impression_id");
            for (String required : columnRequirements) {
                if (!columns.containsKey(required)) {
                    throw new IOException("Missing manifest column: " + required);
                }
            }

            // Turn each CSV row into Fingerprint object
            String line;
            int lineNumber = 1;

            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) continue;

                String[] fields = line.split(",", -1);
                int max = columns.values().stream().mapToInt(Integer::intValue).max().orElse(-1);

                // Check if row contains enough fields
                if (fields.length <= max) {
                    throw new IOException("Malformed manifest line " + lineNumber);
                }

                // Add record to list
                result.add(new Fingerprint(
                        field(fields, columns, "image_path"),
                        field(fields, columns, "class_id"),
                        field(fields, columns, "impression_id"),
                        null));
            }
        }

        // Resolve relative paths against the working directory.
        for (Fingerprint f : result) {
            String newImagePath = "ml_pipeline/" + f.imagePath();
            if (!Files.isRegularFile(Paths.get(newImagePath))) {
                throw new IOException("Image not found: " + newImagePath);
            }
        }

        return result;
    }

    /**
     * Retrieves and trims the value of a named column from a CSV row.
     *
     * @param fields the values parsed from a CSV row
     * @param columns mapping of column names to their zero-based indexes
     * @param name the name of the column to retrieve
     * @return the trimmed value of the specific column
     */
    private static String field(String[] fields, Map<String, Integer> columns, String name) {
        return fields[columns.get(name)].trim();
    }

    /**
     * Formats a string as a quoted CSV field, escaping embedded double quotes
     * by doubling them.
     *
     * @param value the string to format
     * @return the value enclosed in double quotes and escaped for CSV output
     */
    private static String csv(String value) {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }

    static void main(String[] args) throws Exception {
        run();
    }
}