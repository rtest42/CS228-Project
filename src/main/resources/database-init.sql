BEGIN;

DROP TABLE IF EXISTS subjects CASCADE;
DROP TABLE IF EXISTS fingerprints CASCADE;

CREATE TABLE subjects
(
    subject_id TEXT                     NOT NULL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

CREATE TABLE fingerprints
(
    fingerprint_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    subject_id     TEXT                     NOT NULL REFERENCES subjects (subject_id) ON DELETE CASCADE,
    template       BYTEA                    NOT NULL,
    enrolled_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

CREATE INDEX idx_fingerprints_subject_id ON fingerprints (subject_id);

COMMIT;
