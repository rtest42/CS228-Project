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

/** Usage: DegradeRandom window 80 60 40 ...   |   DegradeRandom noise 5 10 20 ... */
public class DegradeRandom {

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

    static int borderMedian(int[] g, int w, int h) {
        int[] b = new int[2 * w + 2 * h];
        int k = 0;
        for (int x = 0; x < w; x++) { b[k++] = g[x]; b[k++] = g[(h - 1) * w + x]; }
        for (int y = 0; y < h; y++) { b[k++] = g[y * w]; b[k++] = g[y * w + w - 1]; }
        Arrays.sort(b);
        return b[b.length / 2];
    }

    static int[] randomWindow(int[] g, int w, int h, double fraction, int fill, Random rnd) {
        double s = Math.sqrt(fraction);
        int kw = (int) Math.round(w * s), kh = (int) Math.round(h * s);
        int x0 = rnd.nextInt(w - kw + 1), y0 = rnd.nextInt(h - kh + 1);
        int[] out = new int[w * h];
        Arrays.fill(out, fill);
        for (int y = y0; y < y0 + kh; y++)
            if (x0 + kw - x0 >= 0) System.arraycopy(g, y * w + x0, out, y * w + x0, x0 + kw - x0);
        return out;
    }

    static int[] addNoise(int[] g, double sigma, Random rnd) {
        int[] out = new int[g.length];
        for (int i = 0; i < g.length; i++)
            out[i] = Math.clamp(Math.round(g[i] + rnd.nextGaussian() * sigma), 0, 255);
        return out;
    }

    static BufferedImage toImage(int[] g, int w, int h) {
        return DegradeBlur.toImage(g, w, h);
    }

    static String label(double v) {
        return v == Math.floor(v) ? String.valueOf((int) v) : String.valueOf(v);
    }

    static void main(String[] args) throws IOException {
        if (args.length < 1 || !(args[0].equals("window") || args[0].equals("noise")))
            throw new IllegalArgumentException("First argument must be 'window' or 'noise'");
        boolean window = args[0].equals("window");

        double[] levels = args.length > 1
                ? Arrays.stream(args).skip(1).mapToDouble(Double::parseDouble).toArray()
                : (window ? new double[]{80, 60, 40, 30, 20, 10}
                : new double[]{5, 10, 20, 30, 40, 60});
        for (double v : levels) {
            if (window && !(v > 0 && v <= 100))
                throw new IllegalArgumentException("Window percent must be in (0, 100]: " + v);
            if (!window && !(v > 0 && v <= 255))
                throw new IllegalArgumentException("Noise sigma must be in (0, 255]: " + v);
        }

        Path src = Path.of("data/CASIA-FingerprintV5(BMP)");
        Path dstRoot = Path.of("data");
        int subjects = 100;
        String prefix = window ? "CASIA-rwin" : "CASIA-noise";

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
                .setTaskName(window ? "Random windows" : "Adding noise")
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
                    int fill = borderMedian(g, w, h);
                    Path rel = src.relativize(p);
                    long pathSeed = rel.toString().replace('\\', '/').hashCode();

                    for (double v : levels) {
                        Random rnd = new Random(pathSeed * 1_000_003L + Double.hashCode(v));
                        int[] out = window ? randomWindow(g, w, h, v / 100.0, fill, rnd)
                                : addNoise(g, v, rnd);
                        Path dst = dstRoot.resolve(prefix + label(v)).resolve(rel);
                        Files.createDirectories(dst.getParent());
                        ImageIO.write(toImage(out, w, h), "bmp", dst.toFile());
                    }
                } catch (IOException e) {
                    throw new RuntimeException(p.toString(), e);
                }
                pb.step();
            });
        }
    }
}