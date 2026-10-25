import java.nio.file.Path;

public class Main {

    public static void main(String[] args) throws Exception {
        System.setProperty("jna.library.path", System.getProperty("user.home") + "/.djl.ai/pytorch/2.7.1-cu128-win-x86_64");
        Path modelPath = Path.of(
                "ml_pipeline/checkpoints/fingerprint_encoder.pt"
        );

        // Load and preprocess both images
        float[] image1 = FingerprintImageLoader.load(
                "ml_pipeline/data/processed/fvc2000/DB1_B/101_1.tif"
        );

        float[] image2 = FingerprintImageLoader.load(
                "ml_pipeline/data/processed/fvc2000/DB1_B/104_5.tif"
        );

        System.out.println("Image 1 pixels: " + image1.length);
        System.out.println("Image 2 pixels: " + image2.length);

        try (FingerprintEncoderService verifier =
                     new FingerprintEncoderService(modelPath)) {

            float distance = verifier.distance(image1, image2);
            boolean match = verifier.isGenuineMatch(image1, image2);

            System.out.println("Fingerprint distance: " + distance);
            System.out.println("Genuine match: " + match);
        }
    }
}
