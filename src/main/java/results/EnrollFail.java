package results;

/**
 * Indicates that enrollment was refused because the fingerprint
 * already matches a different enrolled subject.
 *
 * @param subjectID the existing subject that matched
 * @param score     the SourceAFIS matching score
 */
public record EnrollFail(String subjectID, double score) implements Result {

}
