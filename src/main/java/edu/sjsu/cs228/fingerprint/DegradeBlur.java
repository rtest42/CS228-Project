package edu.sjsu.cs228.fingerprint;

import me.tongfei.progressbar.ConsoleProgressBarConsumer;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarStyle;

import javax.imageio.ImageIO;
import java.awt.image.*;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class DegradeBlur {

    static int[] toGray(BufferedImage img) {
        int w = img.getWidth(), h = img.getHeight();
        int[] g = new int[w * h];
        Raster r = img.getRaster();
        ColorModel cm = img.getColorModel();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int v = r.getSample(x, y, 0);
                g[y * w + x] = (cm instanceof IndexColorModel icm) ? (icm.getRGB(v) & 0xFF) : v;
            }
        return g;
    }

    static float[] kernel(double sigma) {
        int radius = (int) Math.ceil(3 * sigma);
        float[] k = new float[2 * radius + 1];
        double sum = 0;
        for (int i = -radius; i <= radius; i++) {
            double v = Math.exp(-(i * i) / (2 * sigma * sigma));
            k[i + radius] = (float) v;
            sum += v;
        }
        for (int i = 0; i < k.length; i++) k[i] /= (float) sum;
        return k;
    }

    /** Separable Gaussian blur, edge pixels replicated. */
    static int[] blur(int[] g, int w, int h, double sigma) {
        float[] k = kernel(sigma);
        int r = k.length / 2;
        float[] tmp = new float[w * h];

        for (int y = 0; y < h; y++)                    // horizontal pass
            for (int x = 0; x < w; x++) {
                float acc = 0;
                for (int i = -r; i <= r; i++) {
                    int xx = Math.clamp(x + i, 0, w - 1);
                    acc += k[i + r] * g[y * w + xx];
                }
                tmp[y * w + x] = acc;
            }

        int[] out = new int[w * h];
        for (int y = 0; y < h; y++)                    // vertical pass
            for (int x = 0; x < w; x++) {
                float acc = 0;
                for (int i = -r; i <= r; i++) {
                    int yy = Math.clamp(y + i, 0, h - 1);
                    acc += k[i + r] * tmp[yy * w + x];
                }
                out[y * w + x] = Math.clamp(Math.round(acc), 0, 255);
            }
        return out;
    }

    static BufferedImage toImage(int[] g, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster r = out.getRaster();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                r.setSample(x, y, 0, g[y * w + x]);
        return out;
    }

    static String label(double s) {
        return s == Math.floor(s) ? String.valueOf((int) s) : String.valueOf(s);
    }

    static void main(String[] args) throws IOException {
        Path src = Path.of("data/CASIA-FingerprintV5(BMP)");
        Path dstRoot = Path.of("data");
        int subjects = 100;
        double[] sigmas = args.length > 0
                ? Arrays.stream(args).mapToDouble(Double::parseDouble).toArray()
                : new double[]{0.5, 1, 1.5, 2, 3, 4};

        for (double s : sigmas)
            if (!(s > 0 && s <= 20))
                throw new IllegalArgumentException("Sigma must be in (0, 20]: " + s);

        List<Path> files;
        try (Stream<Path> walk = Files.walk(src)) {
            files = walk.filter(p -> p.toString().toLowerCase().endsWith(".bmp"))
                    .filter(p -> {
                        String top = src.relativize(p).getName(0).toString();
                        return top.chars().allMatch(Character::isDigit) && Integer.parseInt(top) < subjects;
                    }).sorted().toList();
        }
        System.out.println(files.size() + " source images");

        try (ProgressBar pb = ProgressBar.builder()
                .setTaskName("Blurring fingerprints")
                .setInitialMax(files.size())
                .setMaxRenderedLength(128)
                .setConsumer(new ConsoleProgressBarConsumer(System.out))
                .setStyle(ProgressBarStyle.UNICODE_BLOCK)
                .build()) {

            files.parallelStream().forEach(p -> {
                try {
                    BufferedImage img = ImageIO.read(p.toFile());
                    int w = img.getWidth(), h = img.getHeight();
                    int[] g = toGray(img);
                    for (double s : sigmas) {
                        Path out = dstRoot.resolve("CASIA-blur" + label(s)).resolve(src.relativize(p));
                        Files.createDirectories(out.getParent());
                        ImageIO.write(toImage(blur(g, w, h, s), w, h), "bmp", out.toFile());
                    }
                } catch (IOException e) {
                    throw new RuntimeException(p.toString(), e);
                }
                pb.step();
            });
        }
    }
}