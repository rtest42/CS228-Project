package fingerprintapp;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

/**
 * Client for sending fingerprint enrollment, verification,
 * and identification requests to the fingerprint server.
 */
public class Client {
    private static final String SERVER_URL = "http://localhost:8080";
    private final HttpClient httpClient;

    /**
     * Creates a client using the default HTTP client.
     */
    public Client() {
        this.httpClient = HttpClient.newHttpClient();
    }

    /**
     * Requests enrollment of a fingerprint for a subject.
     *
     * @param subjectID the identifier of the subject to enroll
     * @param imagePath the path to the fingerprint image file
     */
    public void requestEnroll(String subjectID, String imagePath) {
        this.send("/api/enroll", subjectID, imagePath);
    }

    /**
     * Requests identification of a fingerprint against the enrolled
     * fingerprint gallery.
     *
     * @param imagePath the path to the fingerprint image file
     */
    public void requestIdentify(String imagePath) {
        this.send("/api/identify", null, imagePath);
    }

    /**
     * Requests verification of a fingerprint against a specified subject.
     *
     * @param subjectID the identifier of the subject to verify
     * @param imagePath the path to the fingerprint image file
     */
    public void requestVerify(String subjectID, String imagePath) {
        this.send("/api/verify", subjectID, imagePath);
    }

    /**
     * Helper function to send a fingerprint image for
     * enrollment, identification, or verification.
     *
     * @param endpoint  the API endpoint receiving the request
     * @param subjectID the identifier of the subject
     *                  or {@code null} if performing identification
     * @param imagePath the path to the fingerprint image file
     */
    private void send(String endpoint, String subjectID, String imagePath) {
        try {
            byte[] imageBytes = Files.readAllBytes(Path.of(imagePath));
            String image = Base64.getEncoder().encodeToString(imageBytes);

            String json = subjectID == null
                    ? String.format("{\"image\":\"%s\"}", image)
                    : String.format("{\"subjectID\":%s,\"image\":\"%s\"}",
                    quote(subjectID), image);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(SERVER_URL + endpoint))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response =
                    httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            System.out.printf("[%s -> %d]: %s%n", endpoint, response.statusCode(), response.body());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.printf("Request interrupted: %s%n", e);
        } catch (IOException e) {
            System.err.printf("Network or file error during verification: %s%n", e);
        }
    }

    /**
     * Wraps a string in double quotes for use as a JSON string value,
     * escaping backslashes and double quotes.
     *
     * <p>This is a minimal implementation.
     *
     * @param s the string to quote
     * @return the quoted and escaped string
     */
    private static String quote(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    /**
     * Command-line entry point for sending requests to the fingerprint server.
     *
     * <p>Supported commands:
     * <ul>
     *   <li>{@code enroll <subjectID> <imagePath>}: enrolls a fingerprint image for a subject</li>
     *   <li>{@code verify <subjectID> <imagePath>}: verifies an image against a subject</li>
     *   <li>{@code identify <imagePath>}: identifies the best matching subject for an image</li>
     * </ul>
     *
     * <p>The server must already be running at {@code http://localhost:8080}.
     * Results are printed to standard output.
     *
     * @param args the command and its arguments; prints a usage message
     *             if the arguments are missing or the command is unknown
     */
    static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("""
                    Usage:
                      enroll   <subjectID> <imagePath>
                      verify   <subjectID> <imagePath>
                      identify <imagePath>""");
            System.exit(1);
        }

        Client client = new Client();
        switch (args[0]) {
            case "enroll" -> client.requestEnroll(args[1], args[2]);
            case "verify" -> client.requestVerify(args[1], args[2]);
            case "identify" -> client.requestIdentify(args[1]);
            default -> {
                System.err.printf("Unknown command: %s%n", args[0]);
                System.exit(1);
            }
        }
    }
}
