package commands;

/**
 * Requests verification of a fingerprint against a specific subject.
 *
 * @param subjectID the identifier of the subject to verify
 * @param template  the serialized SourceAFIS probe fingerprint template
 */
public record Verify(String subjectID, byte[] template) implements Command {

}
