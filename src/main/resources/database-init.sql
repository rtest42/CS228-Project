BEGIN;

DROP TABLE IF EXISTS fingerprints CASCADE;

CREATE TABLE fingerprints
(
    fingerprint_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset         TEXT NOT NULL,
    subject_id      TEXT NOT NULL,
    finger_id       TEXT NOT NULL,
    impression      INT NOT NULL,
    template        BYTEA NOT NULL,
    enrolled_at     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    UNIQUE (dataset, subject_id, finger_id, impression)
);

CREATE INDEX idx_fingerprints_dataset_subject ON fingerprints (dataset, subject_id);

COMMIT;
