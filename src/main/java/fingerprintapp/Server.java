package fingerprintapp;

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintTemplate;
import commands.*;
import io.github.cdimascio.dotenv.Dotenv;
import io.javalin.Javalin;
import org.postgresql.ds.PGSimpleDataSource;

import java.util.Base64;
import java.util.Map;

/**
 * Starts the HTTP server for fingerprint enrollment, verification,
 * and identification.
 *
 * <p>Accepts JSON requests, converts submitted fingerprint images
 * into SourceAFIS templates, and delegates operations to
 * {@link Application}.
 */
public class Server {
    /**
     * Prevents instantiation; this class only provides {@link #main(String[])}
     */
    private Server() {
        // Not instantiable; this class only provides main().
    }

    /**
     * Loads configuration, initializes the database connection,
     * registers HTTP endpoints, and starts the server.
     *
     * @param args command-line arguments (currently unused)
     */
    static void main(String[] args) {
        // Retrieve environment variables.
        Dotenv dotenv = Dotenv.load();
        String dbHost = dotenv.get("DB_HOST", "localhost");
        String dbPortStr = dotenv.get("DB_PORT", "5432");
        int dbPort = Integer.parseInt(dbPortStr);
        String dbName = dotenv.get("DB_NAME", "postgres");
        String dbUser = dotenv.get("DB_USER", "postgres");
        String dbPassword = dotenv.get("DB_PASSWORD");
        if (dbPassword == null) {
            throw new IllegalStateException("DB_PASSWORD is not set in .env");
        }

        // Configure the PostgreSQL data source.
        var ds = new PGSimpleDataSource();
        ds.setServerNames(new String[]{dbHost});
        ds.setPortNumbers(new int[]{dbPort});
        ds.setDatabaseName(dbName);
        ds.setUser(dbUser);
        ds.setPassword(dbPassword);

        // Instantiate an application and connect requests to methods.
        try (Application store = new Application(ds)) {
            // Register endpoints that translate HTTP requests into application commands.
            Javalin app = Javalin.create(config -> {
                config.staticFiles.add("/public");

                config.routes.post("/api/enroll", ctx -> {
                    var body = ctx.bodyAsClass(EnrollRequestPayload.class);
                    requireSubject(body.subjectID());
                    var imageBytes = decode(body.image());
                    var templateBytes = createTemplate(imageBytes);
                    var cmd = new Enroll(body.subjectID, templateBytes);
                    ctx.json(store.execute(cmd));
                });

                config.routes.post("/api/verify", ctx -> {
                    var body = ctx.bodyAsClass(VerifyRequestPayload.class);
                    requireSubject(body.subjectID());
                    var imageBytes = decode(body.image());
                    var templateBytes = createTemplate(imageBytes);
                    var cmd = new Verify(body.subjectID, templateBytes);
                    ctx.json(store.execute(cmd));
                });

                config.routes.post("/api/identify", ctx -> {
                    var body = ctx.bodyAsClass(IdentifyRequestPayload.class);
                    var imageBytes = decode(body.image());
                    var templateBytes = createTemplate(imageBytes);
                    var cmd = new Identify(templateBytes);
                    ctx.json(store.execute(cmd));
                });

                config.routes.exception(IllegalArgumentException.class, (e, ctx) ->
                        ctx.status(400).json(Map.of("error", e.getMessage())));
            });

            // Start the server daemon on port 8080 safely.
            app.start(8080);
            Thread.currentThread().join();
        } catch (Exception e) {
            System.err.printf("Database driver or connection failure: %s%n", e.getMessage());
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Decodes a Base64-encoded request.
     *
     * @param base64Str the Base64-encoded data
     * @return the decoded bytes
     * @throws IllegalArgumentException if the input is not valid Base64
     */
    private static byte[] decode(String base64Str) {
        return Base64.getDecoder().decode(base64Str);
    }

    /**
     * Converts encoded fingerprint image bytes into a serialized
     * SourceAFIS fingerprint template.
     *
     * @param imageBytes the encoded fingerprint image
     * @return the serialized fingerprint template
     * @throws IllegalArgumentException if the input is not valid
     */
    private static byte[] createTemplate(byte[] imageBytes) {
        try {
            var fingerprintImage = new FingerprintImage(imageBytes);
            var fingerprintTemplate = new FingerprintTemplate(fingerprintImage);
            return fingerprintTemplate.toByteArray();
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("Invalid fingerprint image", e);
        }
    }

    /**
     * Checks if a subject is provided.
     *
     * @param id the identifier of the subject
     * @throws IllegalArgumentException if the subject is blank
     */
    private static void requireSubject(String id) {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Subject ID is required");
        }
    }

    // Small JSON binding objects matching client payloads
    private record EnrollRequestPayload(String subjectID, String image) {

    }

    private record VerifyRequestPayload(String subjectID, String image) {

    }

    private record IdentifyRequestPayload(String image) {

    }
}
