package results;

/**
 * Contains the outcome of fingerprint identification against the
 * enrolled fingerprint gallery.
 *
 * @param subjectID the identifier of the best matching subject or
 *                  {@code null} if no match was found
 * @param score     the highest SourceAFIS matching score
 */
public record IdentifyOK(String subjectID, double score) implements Result {

}
