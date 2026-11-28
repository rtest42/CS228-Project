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

public class DegradeCrop {

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
        return b[b.length / 2]; // numpy averages the two middle values for even counts
    }

    static BufferedImage centerMask(int[] g, int w, int h, double fraction, int fill) {
        double s = Math.sqrt(fraction);
        int kw = (int) Math.round(w * s), kh = (int) Math.round(h * s);
        int x0 = (w - kw) / 2, y0 = (h - kh) / 2;

        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster r = out.getRaster();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                boolean inside = x >= x0 && x < x0 + kw && y >= y0 && y < y0 + kh;
                r.setSample(x, y, 0, inside ? g[y * w + x] : fill);
            }
        return out;
    }

    static void main(String[] args) throws IOException {
        Path src = Path.of("data/CASIA-FingerprintV5(BMP)").toAbsolutePath().normalize();
        Path dstRoot = Path.of("data").toAbsolutePath().normalize();
        int subjects = 100;
        int[] percents = args.length > 0
                ? Arrays.stream(args).mapToInt(Integer::parseInt).toArray()
                : new int[]{80, 60, 40, 30, 20, 10};

        for (int pct : percents)
            if (pct <= 0 || pct > 100)
                throw new IllegalArgumentException("Percent must be in 1..100: " + pct);

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
                .setTaskName("Degrading fingerprints")
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
                    for (int pct : percents) {
                        Path out = dstRoot.resolve("CASIA-crop" + pct).resolve(src.relativize(p));
                        Files.createDirectories(out.getParent());
                        ImageIO.write(centerMask(g, w, h, pct / 100.0, fill), "bmp", out.toFile());
                    }
                } catch (IOException e) {
                    throw new RuntimeException(p.toString(), e);
                }
                pb.step();
            });
        }
    }
}