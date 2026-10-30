package fingerprintapp;

import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;
import commands.*;
import results.*;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

/**
 * Provides fingerprint enrollment, verification, and identification.
 *
 * <p>Uses a database to store enrolled fingerprint templates
 * and SourceAFIS to compare fingerprints.
 */
public class Application implements AutoCloseable {
    private final double threshold;
    private final DataSource dataSource;

    /**
     * Creates an application with the specified data source.
     *
     * @param dataSource the data source used to access stored subjects
     *                   and fingerprint templates
     */
    public Application(DataSource dataSource) {
        this.dataSource = dataSource;
        this.threshold = 40.0;
    }

    /**
     * Executes a fingerprint command.
     *
     * @param command the command to execute
     * @return the result of the command
     * @throws IllegalArgumentException if the command type is unsupported
     * @throws IllegalStateException    if a database operation fails
     */
    public Result execute(Command command) {
        return switch (command) {
            case Enroll enroll -> this.executeEnroll(enroll);
            case Verify verify -> this.executeVerify(verify);
            case Identify identify -> this.executeIdentify(identify);
            default -> throw new IllegalArgumentException(
                    "Unsupported command: " + command.getClass().getName()
            );
        };
    }

    /**
     * Enrolls a fingerprint template for a subject.
     *
     * <p>An identical existing template is treated as an idempotent retry.
     *
     * @param enroll the enrollment request
     * @return the enrollment outcome
     */
    private Result executeEnroll(Enroll enroll) {
        String createSubject = "INSERT INTO subjects(subject_id) VALUES (?) ON CONFLICT DO NOTHING";
        String insertTemplate = "INSERT INTO fingerprints(subject_id, template) VALUES (?, ?)";

        // Reject enrollment if this fingerprint already matches any enrolled subject.
        Result existing = this.executeIdentify(new Identify(enroll.template()));
        if (existing instanceof IdentifyOK(String subjectID, double score)
                && subjectID != null) {
            return new EnrollFail(subjectID, score);
        }

        try (Connection c = this.dataSource.getConnection()) {
            c.setAutoCommit(false);

            try {
                // Create the subject if this is its first enrollment.
                try (PreparedStatement stmt = c.prepareStatement(createSubject)) {
                    stmt.setString(1, enroll.subjectID());
                    stmt.executeUpdate();
                }

                // Enroll the new fingerprint.
                try (PreparedStatement stmt = c.prepareStatement(insertTemplate)) {
                    stmt.setString(1, enroll.subjectID());
                    stmt.setBytes(2, enroll.template());
                    stmt.executeUpdate();
                }

                c.commit();
                return new EnrollOK(enroll.subjectID());
            } catch (SQLException | RuntimeException e) {
                try {
                    c.rollback();
                } catch (SQLException rollbackError) {
                    e.addSuppressed(rollbackError);
                }

                throw e;
            } finally {
                c.setAutoCommit(true);
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Database error", e);
        }
    }

    /**
     * Verifies that the selected fingerprint matches the selected subject.
     *
     * @param verify the verification request
     * @return the verification outcome
     */
    private Result executeVerify(Verify verify) {
        String selectTemplate = "SELECT template FROM fingerprints WHERE subject_id = ?";

        try (Connection c = this.dataSource.getConnection();
             PreparedStatement sel = c.prepareStatement(selectTemplate)) {
            sel.setString(1, verify.subjectID());

            var probe = new FingerprintMatcher(
                    new FingerprintTemplate(verify.template())
            );
            double bestScore = 0.0;

            try (ResultSet rs = sel.executeQuery()) {
                // A subject may have multiple enrolled fingerprints.
                while (rs.next()) {
                    var candidate = new FingerprintTemplate(rs.getBytes(1));
                    // Keep track of scores for verification outcomes.
                    bestScore = Math.max(bestScore, probe.match(candidate));
                }
            }

            return new VerifyOK(bestScore >= this.threshold, bestScore);
        } catch (SQLException e) {
            throw new IllegalStateException("Database error", e);
        }
    }

    /**
     * Identifies the most plausible subject given by
     * the selected fingerprint.
     *
     * @param identify the identification request
     * @return the identification outcome
     */
    private Result executeIdentify(Identify identify) {
        String selectAll = "SELECT subject_id, template FROM fingerprints";

        var probe = new FingerprintMatcher(
                new FingerprintTemplate(identify.template())
        );
        String bestID = null;
        double bestScore = 0.0;

        try (Connection c = this.dataSource.getConnection();
             PreparedStatement sel = c.prepareStatement(selectAll)) {
            sel.setFetchSize(512);

            try (ResultSet rs = sel.executeQuery()) {
                // Compare the probe against every enrolled fingerprint.
                while (rs.next()) {
                    var candidate = new FingerprintTemplate(rs.getBytes(2));
                    double score = probe.match(candidate);

                    // Keep the subject associated with the highest-scoring template.
                    if (score > bestScore) {
                        bestScore = score;
                        bestID = rs.getString(1);
                    }
                }
            }

            // Return no identity if the best score doesn't meet the threshold.
            if (bestScore < this.threshold) {
                bestID = null;
            }

            return new IdentifyOK(bestID, bestScore);
        } catch (SQLException e) {
            throw new IllegalStateException("Database error", e);
        }
    }

    /**
     * Releases the resources owned by this application.
     *
     * <p>Closes the data source if it implements {@link AutoCloseable}.
     *
     * @throws Exception if closing the data source fails
     */
    @Override
    public void close() throws Exception {
        if (this.dataSource instanceof AutoCloseable ac) {
            ac.close();
        }
    }
}
