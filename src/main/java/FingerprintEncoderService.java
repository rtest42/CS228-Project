
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.nio.file.Path;

public class FingerprintEncoderService implements AutoCloseable {

    private final Model model;
    private final Predictor<NDList, float[]> predictor;
    private final NDManager manager;

    public FingerprintEncoderService(Path modelPath) throws Exception {
        manager = NDManager.newBaseManager();

        model = Model.newInstance("fingerprint-encoder", "PyTorch");
        model.load(modelPath);

        Translator<NDList, float[]> translator =
            new Translator<>() {
                @Override
                public NDList processInput(
                        TranslatorContext ctx, NDList input) {
                    return input;
                }

                @Override
                public float[] processOutput(
                        TranslatorContext ctx, NDList output) {
                    return output.getFirst().toFloatArray();
                }
            };

        predictor = model.newPredictor(translator);
    }

    public float[] embed(float[] pixels) throws TranslateException {
        if (pixels.length != 128 * 128) {
            throw new IllegalArgumentException(
                "Expected exactly 16384 grayscale pixels."
            );
        }

        var image = manager.create(
            pixels,
            new Shape(1, 128, 128)
        );

        return predictor.predict(new NDList(image));
    }

    public float distance(float[] image1, float[] image2)
            throws TranslateException {

        float[] a = embed(image1);
        float[] b = embed(image2);

        if (a.length != b.length) {
            throw new IllegalStateException(
                "Embedding dimensions do not match."
            );
        }

        double sum = 0.0;

        for (int i = 0; i < a.length; i++) {
            double difference = a[i] - b[i];
            sum += difference * difference;
        }

        return (float) Math.sqrt(sum);
    }

    public boolean isGenuineMatch(float[] image1, float[] image2)
            throws TranslateException {
        return distance(image1, image2) <= 0.389967f;
    }

    @Override
    public void close() {
        predictor.close();
        model.close();
        manager.close();
    }
}
