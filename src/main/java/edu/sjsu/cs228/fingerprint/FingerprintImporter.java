package edu.sjsu.cs228.fingerprint;

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintTemplate;
import io.github.cdimascio.dotenv.Dotenv;
import me.tongfei.progressbar.ConsoleProgressBarConsumer;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarStyle;

import javax.sql.DataSource;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.*;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class FingerprintImporter {

    @FunctionalInterface
    public interface ConnectionSupplier {
        Connection get() throws SQLException;
    }

    public record DatasetConfig(
            String datasetName,
            Path rootDir,
            double dpi,
            Set<String> allowedExtensions,
            Function<Path, FingerprintImporter.FingerprintMetadata> metadataExtractor
    ) {}

    public record FingerprintMetadata(
            String subjectId,
            String fingerId,
            int impression
    ) {}

    private record FingerprintRecord(
            String dataset,
            FingerprintMetadata meta,
            byte[] templateBytes
    ) {}

    private static final String INSERT_SQL = """
            INSERT INTO fingerprints (dataset, subject_id, finger_id, impression, template)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (dataset, subject_id, finger_id, impression) DO NOTHING
            """;

    private final ConnectionSupplier connectionSupplier;
    private final int batchSize;
    private static final int DEFAULT_BATCH_SIZE = 512;

    // Constructor 1: Use with a connection pool (e.g., HikariDataSource)
    public FingerprintImporter(DataSource dataSource) {
        this(dataSource::getConnection, DEFAULT_BATCH_SIZE);
    }

    // Constructor 2: Use with JDBC URL/credentials
    public FingerprintImporter(String jdbcUrl, String user, String password) {
        this(() -> DriverManager.getConnection(jdbcUrl, user, password), DEFAULT_BATCH_SIZE);
    }

    // Constructor 3: Full control over ConnectionSupplier and batchSize
    public FingerprintImporter(ConnectionSupplier connectionSupplier, int batchSize) {
        this.connectionSupplier = Objects.requireNonNull(connectionSupplier);
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }
        this.batchSize = batchSize;
    }

    /**
     * Helper to create a path parser using named regex groups:
     * Required: (?<subject>...) and (?<impression>...)
     * Optional: (?<finger>...) - defaults to "0" if omitted
     */
    public Function<Path, FingerprintMetadata> createRegexParser(String regex) {
        Pattern pattern = Pattern.compile(regex, Pattern.CASE_INSENSITIVE);
        boolean hasFingerGroup = regex.contains("(?<finger>");

        return relativePath -> {
            String normalized = relativePath.toString().replace('\\', '/');
            Matcher matcher = pattern.matcher(normalized);
            if (!matcher.matches()) {
                throw new IllegalArgumentException("Path did not match pattern: " + normalized);
            }

            String subjectId = matcher.group("subject");
            String fingerId = hasFingerGroup ? matcher.group("finger") : "0";
            int impression = Integer.parseInt(matcher.group("impression"));

            return new FingerprintMetadata(subjectId, fingerId, impression);
        };
    }

    public void importDataset(DatasetConfig config) throws Exception {
        long start = System.currentTimeMillis();

        List<Path> files = collectFiles(config);
        System.out.printf("[%s] Found %d images. Extracting templates at %.0f DPI...%n",
                config.datasetName(), files.size(), config.dpi());

        List<FingerprintRecord> records;

        try (ProgressBar pb = ProgressBar.builder()
                .setTaskName("Extracting templates")
                .setInitialMax(files.size())
                .setMaxRenderedLength(128)
                .setConsumer(new ConsoleProgressBarConsumer(System.out))
                .setStyle(ProgressBarStyle.UNICODE_BLOCK)
                .build()) {
            records = files.parallelStream()
                    .map(path -> {
                        FingerprintRecord record = extractRecord(config, path);

                        synchronized (pb) {
                            pb.step();
                        }

                        return record;
                    })
                    .toList();
        }

        insertBatch(records);

        System.out.printf("[%s] Imported %d records in %.2f s.%n",
                config.datasetName(), records.size(), (System.currentTimeMillis() - start) / 1000.0);
    }

    private List<Path> collectFiles(DatasetConfig config) throws Exception {
        try (Stream<Path> walk = Files.walk(config.rootDir())) {
            return walk
                    .filter(Files::isRegularFile)
                    .filter(p -> hasAllowedExtension(p, config.allowedExtensions()))
                    .toList();
        }
    }

    private FingerprintRecord extractRecord(DatasetConfig config, Path fullPath) {
        Path relativePath = config.rootDir().relativize(fullPath);
        FingerprintMetadata meta = config.metadataExtractor().apply(relativePath);

        try {
            byte[] imageBytes = Files.readAllBytes(fullPath);
            FingerprintImage image = new FingerprintImage(
                    imageBytes,
                    new FingerprintImageOptions().dpi(config.dpi())
            );
            byte[] templateBytes = new FingerprintTemplate(image).toByteArray();
            return new FingerprintRecord(config.datasetName(), meta, templateBytes);
        } catch (Exception e) {
            throw new RuntimeException("Failed processing file: " + fullPath, e);
        }
    }

    private void insertBatch(List<FingerprintRecord> records) throws SQLException {
        try (Connection conn = connectionSupplier.get();
             ProgressBar pb = ProgressBar.builder()
                     .setTaskName("Inserting into PostgreSQL")
                     .setInitialMax(records.size())
                     .setMaxRenderedLength(128)
                     .setConsumer(new ConsoleProgressBarConsumer(System.out))
                     .setStyle(ProgressBarStyle.UNICODE_BLOCK)
                     .build()) {
            conn.setAutoCommit(false);

            try (PreparedStatement ps = conn.prepareStatement(INSERT_SQL)) {
                int pending = 0;

                for (FingerprintRecord rec : records) {
                    ps.setString(1, rec.dataset());
                    ps.setString(2, rec.meta().subjectId());
                    ps.setString(3, rec.meta().fingerId());
                    ps.setInt(4, rec.meta().impression());
                    ps.setBytes(5, rec.templateBytes());
                    ps.addBatch();

                    if (++pending == batchSize) {
                        ps.executeBatch();
                        pb.stepBy(pending);
                        pending = 0;
                    }
                }
                if (pending > 0) {
                    ps.executeBatch();
                    pb.stepBy(pending);
                }
                conn.commit();
            } catch (SQLException e) {
                conn.rollback();
                throw e;
            }
        }
    }

    private boolean hasAllowedExtension(Path path, Set<String> extensions) {
        String name = path.getFileName().toString().toLowerCase();
        int dot = name.lastIndexOf('.');
        return dot != -1 && extensions.contains(name.substring(dot + 1));
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

        String JDBC_URL =  "jdbc:postgresql://" + dbHost + ":" + dbPort + "/" + dbName + "?reWriteBatchedInserts=true";
        var importer = new FingerprintImporter(JDBC_URL, dbUser, dbPassword);

        // CASIA-FingerprintV5: captures "000" as subject, "L0"/"R3" as fingerId, and "0" as impression
        Map<String, String> options = importer.parseArgs(args);
        var casiaConfig = new FingerprintImporter.DatasetConfig(
                options.getOrDefault("dataset", "CASIA-FingerprintV5"),
                Path.of(options.getOrDefault("path", "data/CASIA-FingerprintV5(BMP)")).toAbsolutePath().normalize(),
                Double.parseDouble(options.getOrDefault("dpi", "512")),
                Arrays.stream(options.getOrDefault("extensions", "bmp").split(","))
                        .map(String::trim)
                        .map(String::toLowerCase)
                        .filter(s -> !s.isEmpty())
                        .collect(Collectors.toUnmodifiableSet()),
                importer.createRegexParser(options.getOrDefault("pattern",
                        ".*/(?<subject>\\d{3})_(?<finger>[LR]\\d)_(?<impression>\\d+)\\.bmp$"
                ))
        );

        importer.importDataset(casiaConfig);
    }

    private Map<String, String> parseArgs(String[] args) {
        Map<String, String> options = new HashMap<>();

        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                String key = args[i].substring(2);

                if (i + 1 >= args.length || args[i + 1].startsWith("--")) {
                    throw new IllegalArgumentException("Missing value for --" + key);
                }

                options.put(key, args[++i]);
            }
        }

        return options;
    }
}
