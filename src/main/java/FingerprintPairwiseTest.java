
import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintTemplate;
import com.machinezoo.sourceafis.FingerprintMatcher;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class FingerprintPairwiseTest {

    private static final int THREADS =
            Runtime.getRuntime().availableProcessors();

    // Adjust if your manifest paths are relative to another directory.
    private static final Path MANIFEST = Paths.get("ml_pipeline/manifest.csv");
    private static final Path OUTPUT = Paths.get("pairwise_scores.csv");

    record Fingerprint(
            String imagePath,
            String dataset,
            String database,
            String fingerId,
            String impressionId,
            FingerprintTemplate template) {}

    public static void main(String[] args) throws Exception {
        long start = System.nanoTime();

        System.out.println("Available processors: " + THREADS);

        List<Fingerprint> fingerprints = loadManifest(MANIFEST);
        System.out.println("Manifest entries: " + fingerprints.size());

        // Stage 1: Feature extraction, parallelized across CPUs.
        ExecutorService pool = Executors.newFixedThreadPool(THREADS);

        try {
            List<Future<Fingerprint>> futures = new ArrayList<>();

            for (Fingerprint f : fingerprints) {
                futures.add(pool.submit(() -> {
                    Path path = Paths.get(f.imagePath());

                    FingerprintTemplate template =
                            new FingerprintTemplate(
                                    new FingerprintImage(
                                            Files.readAllBytes(path)));

                    return new Fingerprint(
                            f.imagePath(),
                            f.dataset(),
                            f.database(),
                            f.fingerId(),
                            f.impressionId(),
                            template);
                }));
            }

            List<Fingerprint> ready = new ArrayList<>();
            for (Future<Fingerprint> future : futures) {
                // Propagates extraction errors rather than silently
                // producing incomplete evaluation results.
                ready.add(future.get());
            }

            System.out.println("Templates extracted: " + ready.size());

            // Stage 2: Every unordered pair, exactly once.
            long totalPairs =
                    (long) ready.size() * (ready.size() - 1) / 2;

            System.out.println("Total pairs: " + totalPairs);

            try (BufferedWriter writer = Files.newBufferedWriter(
                    OUTPUT, StandardCharsets.UTF_8)) {

                writer.write(
                        "image_path_1,image_path_2,"
                                + "finger_id_1,finger_id_2,"
                                + "impression_id_1,impression_id_2,"
                                + "label,score");
                writer.newLine();

                // Chunk work to avoid creating hundreds of thousands
                // of individual tasks.
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
                            FingerprintMatcher matcher =
                                    new FingerprintMatcher(a.template());

                            for (int j = i1 + 1; j < ready.size(); j++) {
                                Fingerprint b = ready.get(j);

                                double score = matcher.match(b.template());

                                int label = a.fingerId().equals(b.fingerId())
                                        ? 1 : 0;

                                rows.add(csv(a.imagePath()) + ","
                                        + csv(b.imagePath()) + ","
                                        + csv(a.fingerId()) + ","
                                        + csv(b.fingerId()) + ","
                                        + csv(a.impressionId()) + ","
                                        + csv(b.impressionId()) + ","
                                        + label + ","
                                        + score);
                            }
                        }
                        return rows;
                    });
                }

                long completed = 0;

                // ExecutorCompletionService allows results to be written
                // as chunks finish, instead of holding all rows in RAM.
                ExecutorCompletionService<List<String>> completion =
                        new ExecutorCompletionService<>(pool);

                for (Callable<List<String>> task : tasks) {
                    completion.submit(task);
                }

                for (int i = 0; i < tasks.size(); i++) {
                    List<String> rows = completion.take().get();

                    for (String row : rows) {
                        writer.write(row);
                        writer.newLine();
                    }

                    completed += rows.size();
                    if (completed % 20000 < rows.size()
                            || completed == totalPairs) {
                        System.out.printf(
                                "Compared %,d / %,d pairs%n",
                                completed, totalPairs);
                    }
                }
            }

            System.out.println("Output: " + OUTPUT.toAbsolutePath());

        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(1, TimeUnit.MINUTES)) {
                pool.shutdownNow();
            }
        }

        System.out.printf("Elapsed: %.2f seconds%n",
                (System.nanoTime() - start) / 1e9);
    }

    private static List<Fingerprint> loadManifest(Path manifest)
            throws IOException {

        List<Fingerprint> result = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(
                manifest, StandardCharsets.UTF_8)) {

            String header = reader.readLine();
            if (header == null) {
                throw new IOException("Manifest is empty");
            }

            Map<String, Integer> columns = Arrays.stream(
                            header.replace("\uFEFF", "").split(",", -1))
                    .map(String::trim)
                    .collect(Collectors.toMap(
                            s -> s,
                            new java.util.function.Function<String, Integer>() {
                                private int index = 0;
                                @Override
                                public Integer apply(String ignored) {
                                    return index++;
                                }
                            },
                            (a, b) -> a,
                            LinkedHashMap::new));

            for (String required : List.of(
                    "image_path", "dataset", "database",
                    "finger_id", "impression_id")) {
                if (!columns.containsKey(required)) {
                    throw new IOException(
                            "Missing manifest column: " + required);
                }
            }

            String line;
            int lineNumber = 1;

            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) continue;

                // Suitable for the shown manifest, which has no quoted
                // commas. Use a CSV library for arbitrary CSV quoting.
                String[] fields = line.split(",", -1);

                int max = columns.values().stream()
                        .mapToInt(Integer::intValue).max().orElse(-1);

                if (fields.length <= max) {
                    throw new IOException(
                            "Malformed manifest line " + lineNumber);
                }

                result.add(new Fingerprint(
                        field(fields, columns, "image_path"),
                        field(fields, columns, "dataset"),
                        field(fields, columns, "database"),
                        field(fields, columns, "finger_id"),
                        field(fields, columns, "impression_id"),
                        null));
            }
        }

        // Resolve relative paths against the working directory.
        for (Fingerprint f : result) {
            if (!Files.isRegularFile(Paths.get(f.imagePath()))) {
                throw new IOException(
                        "Image not found: " + f.imagePath());
            }
        }

        return result;
    }

    private static String field(
            String[] fields, Map<String, Integer> columns, String name) {
        return fields[columns.get(name)].trim();
    }

    private static String csv(String value) {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }
}
