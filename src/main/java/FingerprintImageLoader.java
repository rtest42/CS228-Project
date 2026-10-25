import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class FingerprintImageLoader {

    public static float[] load(String path) throws IOException {
        BufferedImage original = ImageIO.read(new File(path));

        if (original == null) {
            throw new IOException("Could not load image: " + path);
        }

        // Resize to 128 x 128 grayscale
        BufferedImage resized = new BufferedImage(
                128, 128, BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
        );
        g.drawImage(original, 0, 0, 128, 128, null);
        g.dispose();

        // Convert pixels to float values normalized to [0, 1]
        float[] pixels = new float[128 * 128];

        for (int y = 0; y < 128; y++) {
            for (int x = 0; x < 128; x++) {
                int gray = resized.getRaster().getSample(x, y, 0);

                pixels[y * 128 + x] = gray / 255.0f;
            }
        }

        return pixels;
    }
}
