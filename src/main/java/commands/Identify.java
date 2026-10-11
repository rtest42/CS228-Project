package commands;

/**
 * Requests identification of a subject by matching a fingerprint
 * against the enrolled fingerprint gallery.
 *
 * @param template the serialized SourceAFIS fingerprint template to identify
 */
public record Identify(byte[] template) implements Command {

}
