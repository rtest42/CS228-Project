package edu.sjsu.cs228.fingerprint;

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintTemplate;

import javax.sql.DataSource;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
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
            Function<Path, FingerprintMetadata> metadataExtractor
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

    // Constructor 1: Use with a connection pool (e.g., HikariDataSource)
    public FingerprintImporter(DataSource dataSource) {
        this(dataSource::getConnection, 512);
    }

    // Constructor 2: Use with JDBC URL/credentials
    public FingerprintImporter(String jdbcUrl, String user, String password) {
        this(() -> DriverManager.getConnection(jdbcUrl, user, password), 512);
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

    public int importDataset(DatasetConfig config) throws Exception {
        long start = System.currentTimeMillis();

        List<Path> files = collectFiles(config);
        System.out.printf("[%s] Found %d images. Extracting templates at %.0f DPI...%n",
                config.datasetName(), files.size(), config.dpi());

        List<FingerprintRecord> records = files.parallelStream()
                .map(path -> extractRecord(config, path))
                .toList();

        insertBatch(records);

        System.out.printf("[%s] Imported %d records in %.2f s.%n",
                config.datasetName(), records.size(), (System.currentTimeMillis() - start) / 1000.0);

        return records.size();
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
             PreparedStatement ps = conn.prepareStatement(INSERT_SQL)) {

            conn.setAutoCommit(false);
            int count = 0;

            for (FingerprintRecord rec : records) {
                ps.setString(1, rec.dataset());
                ps.setString(2, rec.meta().subjectId());
                ps.setString(3, rec.meta().fingerId());
                ps.setInt(4, rec.meta().impression());
                ps.setBytes(5, rec.templateBytes());
                ps.addBatch();

                if (++count % batchSize == 0) {
                    ps.executeBatch();
                }
            }
            ps.executeBatch();
            conn.commit();
        }
    }

    private boolean hasAllowedExtension(Path path, Set<String> extensions) {
        String name = path.getFileName().toString().toLowerCase();
        int dot = name.lastIndexOf('.');
        return dot != -1 && extensions.contains(name.substring(dot + 1));
    }

    static void main(String[] args) throws Exception {
        var importer = new FingerprintImporter(
                "jdbc:postgresql://localhost:5432/biometrics?reWriteBatchedInserts=true",
                "postgres",
                "your_password"
        );

        // CASIA-FingerprintV5: captures "000" as subject, "L0"/"R3" as fingerId, and "0" as impression
        var casiaConfig = new FingerprintImporter.DatasetConfig(
                "CASIA-V5",
                Path.of("/data/CASIA-FingerprintV5"),
                512.0,
                Set.of("bmp"),
                importer.createRegexParser(
                        ".*/(?<subject>\\d{3})_(?<finger>[LR]\\d)_(?<impression>\\d+)\\.bmp$"
                )
        );

        importer.importDataset(casiaConfig);
    }
}
