package commands;

/**
 * Requests enrollment of a fingerprint for a subject.
 *
 * @param subjectID the identifier of the subject
 * @param template  the serialized SourceAFIS fingerprint template
 */
public record Enroll(String subjectID, byte[] template) implements Command {

}