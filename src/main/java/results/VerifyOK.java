package results;

/**
 * Contains the outcome of fingerprint verification.
 *
 * @param verified whether the fingerprint meets the matching threshold
 * @param score    the SourceAFIS matching score
 */
public record VerifyOK(boolean verified, double score) implements Result {

}
