package edu.sjsu.cs228.fingerprint;

import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;
import io.github.cdimascio.dotenv.Dotenv;
import me.tongfei.progressbar.ConsoleProgressBarConsumer;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarStyle;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

public class FingerprintEvaluator {

    private static final int BUCKETS = 2001; // integer score buckets 0..2000 (higher scores clamp)

    private final FingerprintTemplate[] templates;
    private final int[] identity; // same value => same finger (genuine pair)

    public FingerprintEvaluator(String jdbcUrl, String user, String password, String datasetName) throws SQLException {
        List<byte[]> blobs = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        Map<String, Integer> identityMap = new HashMap<>();

        // subject_id < '100' to limit to 100 subjects instead of the entire 500
        // cuts execution time by 5^2=25x
        String sql = """
                SELECT dataset, subject_id, finger_id, template
                FROM fingerprints
                WHERE dataset = ? AND subject_id < '100'
                ORDER BY subject_id, finger_id, impression
                """;

        try (Connection conn = DriverManager.getConnection(jdbcUrl, user, password)) {
            conn.setAutoCommit(false); // required for cursor-based fetching in PostgreSQL
            try (PreparedStatement st = conn.prepareStatement(sql, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY)) {
                st.setFetchSize(1000);
                st.setString(1, datasetName);
                try (ResultSet rs = st.executeQuery()) {
                    while (rs.next()) {
                        String key = rs.getString(1) + "|" + rs.getString(2) + "|" + rs.getString(3);
                        ids.add(identityMap.computeIfAbsent(key, k -> identityMap.size()));
                        blobs.add(rs.getBytes(4));
                    }
                }
            }
        }

        if (blobs.isEmpty()) {
            throw new IllegalStateException("No templates found for dataset: " + datasetName);
        }

        this.identity = ids.stream().mapToInt(Integer::intValue).toArray();
        this.templates = blobs.parallelStream()
                .map(FingerprintTemplate::new)   // deserialize
                .toArray(FingerprintTemplate[]::new);

        System.out.printf("[%s] Loaded %d templates, %d distinct fingers.%n", datasetName, templates.length, identityMap.size());
    }

    public record Histograms(long[] genuine, long[] impostor) {}

    public Histograms computeScoreHistograms() throws Exception {
        int n = templates.length;
        int threads = Runtime.getRuntime().availableProcessors();
        AtomicInteger nextRow = new AtomicInteger();
        long[][] genLocal = new long[threads][BUCKETS];
        long[][] impLocal = new long[threads][BUCKETS];

        long totalPairs = (long) n * (n - 1) / 2;

        try (ProgressBar pb = ProgressBar.builder()
                .setTaskName("Matching fingerprints")
                .setInitialMax(totalPairs)
                .setMaxRenderedLength(128)
                .setConsumer(new ConsoleProgressBarConsumer(System.out))
                .setStyle(ProgressBarStyle.UNICODE_BLOCK)
                .build();
             ExecutorService pool = Executors.newFixedThreadPool(threads)) {

            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final long[] g = genLocal[t], m = impLocal[t];
                futures.add(pool.submit(() -> {
                    int i;
                    while ((i = nextRow.getAndIncrement()) < n - 1) {
                        FingerprintMatcher matcher = new FingerprintMatcher(templates[i]);
                        for (int j = i + 1; j < n; j++) {
                            int bucket = (int) Math.min(matcher.match(templates[j]), BUCKETS - 1);
                            if (identity[i] == identity[j]) g[bucket]++; else m[bucket]++;
                        }
                        synchronized (pb) { pb.stepBy(n - 1 - i); }
                    }
                }));
            }
            for (Future<?> f : futures) f.get(); // propagates worker exceptions
        }

        long[] gen = new long[BUCKETS], imp = new long[BUCKETS];
        for (int t = 0; t < threads; t++)
            for (int b = 0; b < BUCKETS; b++) { gen[b] += genLocal[t][b]; imp[b] += impLocal[t][b]; }
        return new Histograms(gen, imp);
    }

    public static void printConfusionMatrix(Histograms h, int threshold) {
        long tp = 0, fn = 0, fp = 0, tn = 0;
        for (int b = 0; b < BUCKETS; b++) {
            if (b >= threshold) { tp += h.genuine()[b]; fp += h.impostor()[b]; }
            else                { fn += h.genuine()[b]; tn += h.impostor()[b]; }
        }
        double fmr  = fp / (double) (fp + tn);   // false match rate
        double fnmr = fn / (double) (fn + tp);   // false non-match rate

        System.out.printf("%nThreshold = %d (match if score >= threshold)%n", threshold);

        System.out.println("+----------------------+------------------+---------------------+");
        System.out.printf("| %-20s | %16s | %19s |%n",
                "", "Predicted match", "Predicted non-match");
        System.out.println("+----------------------+------------------+---------------------+");

        System.out.printf("| %-20s | %,16d | %,19d |%n",
                "Actual genuine", tp, fn);

        System.out.printf("| %-20s | %,16d | %,19d |%n",
                "Actual impostor", fp, tn);

        System.out.println("+----------------------+------------------+---------------------+");

        System.out.printf("FMR  = %.4f%%  |  FNMR = %.4f%%  |  TPR = %.4f%%%n",
                fmr * 100, fnmr * 100, (1 - fnmr) * 100);
    }

    /** Sweeps thresholds and reports where FMR ~= FNMR. */
    public static void printEer(Histograms h) {
        long totalG = Arrays.stream(h.genuine()).sum();
        long totalI = Arrays.stream(h.impostor()).sum();
        System.out.printf("Genuine pairs: %,d   Impostor pairs: %,d%n", totalG, totalI);

        long gBelow = 0, iBelow = 0;
        double bestGap = Double.MAX_VALUE, eer = 0; int eerT = 0;
        for (int t = 0; t < BUCKETS; t++) {
            // at threshold t: below = buckets < t
            double fnmr = gBelow / (double) totalG;
            double fmr  = (totalI - iBelow) / (double) totalI;
            if (Math.abs(fmr - fnmr) < bestGap) { bestGap = Math.abs(fmr - fnmr); eer = (fmr + fnmr) / 2; eerT = t; }
            gBelow += h.genuine()[t];
            iBelow += h.impostor()[t];
        }
        System.out.printf("EER ≈ %.4f%% at threshold %d%n", eer * 100, eerT);
    }

    static void saveHistograms(Histograms h, Path file) throws IOException {
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (BufferedWriter w = Files.newBufferedWriter(file)) {
            w.write("score,genuine,impostor\n");
            for (int b = 0; b < BUCKETS; b++)
                if (h.genuine()[b] != 0 || h.impostor()[b] != 0)
                    w.write(b + "," + h.genuine()[b] + "," + h.impostor()[b] + "\n");
        }
    }

    static void main(String[] args) throws Exception {
        Dotenv dotenv = Dotenv.load();
        String dbHost = dotenv.get("DB_HOST", "localhost");
        String dbPortStr = dotenv.get("DB_PORT", "5432");
        int dbPort = Integer.parseInt(dbPortStr);
        String dbName = dotenv.get("DB_NAME", "fingerprints");
        String dbUser = dotenv.get("DB_USER", "postgres");
        String dbPassword = dotenv.get("DB_PASSWORD");
        if (dbPassword == null) throw new IllegalStateException("DB_PASSWORD is not set in .env");

        String JDBC_URL =  "jdbc:postgresql://" + dbHost + ":" + dbPort + "/" + dbName;

        String dataset = args.length > 0 ? args[0] : "CASIA-FingerprintV5";
        var eval = new FingerprintEvaluator(JDBC_URL, dbUser, dbPassword, dataset);
        Histograms h = eval.computeScoreHistograms();

        printEer(h);
        printConfusionMatrix(h, 40); // SourceAFIS docs: 40 ≈ 0.01% FMR
        printConfusionMatrix(h, 30);
        printConfusionMatrix(h, 60);

        saveHistograms(h, Path.of("results", dataset + ".csv"));
    }
}
