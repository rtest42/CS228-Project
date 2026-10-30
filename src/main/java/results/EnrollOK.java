package results;

/**
 * Indicates that a fingerprint was successfully enrolled for a subject.
 *
 * @param subjectID the identifier of the subject whose fingerprint
 *                  was enrolled
 */
public record EnrollOK(String subjectID) implements Result {

}
