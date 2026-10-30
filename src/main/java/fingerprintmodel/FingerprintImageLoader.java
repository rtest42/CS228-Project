package fingerprintmodel;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * Loads a fingerprint image from a file and converts it into a numerical
 * array that can be used by a fingerprint recognition or machine learning
 * model.
 */
public class FingerprintImageLoader {
    /**
     *  Loads a fingerprint image from the specified file path, resizes it
     *  to 256 x 256 pixels, converts it to grayscale, and normalizes its
     *  pixel values to the range [0.0, 1.0].
     *
     * @param path the file path of the fingerprint image to load
     * @return a float array containing 256*256 normalized grayscale pixel
     * values, each between 0.0 and 1.0
     * @throws IOException if the image cannot be read or the file does not
     * contain a supported image format
     */
    public static float[] load(String path) throws IOException {
        final int width = 256;
        final int height = 256;

        // Read image file and create BufferedImage object.
        // Throw exception if decode failed.
        BufferedImage original = ImageIO.read(new File(path));
        if (original == null) {
            throw new IOException("Could not load image: " + path);
        }

        // Resize to 256 x 256 grayscale
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        // Draw original image onto new grayscale image.
        // Use bilinear interpolation for smoother resizing.
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, width, height, null);
        g.dispose();

        // Convert pixels to float values normalized to [0, 1]
        float[] pixels = new float[width * height];

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int gray = resized.getRaster().getSample(x, y, 0);

                pixels[y * width + x] = gray / 255.0f;
            }
        }

        // Flattened array of normalized pixel values.
        return pixels;
    }
}
