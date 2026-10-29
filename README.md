# CS228 Project - Fingerprint Matching Portal

## Overview

A Java web service for fingerprint enrollment, verification (1:1), and identification (1:N). Fingerprint images are converted to templates and compared with [SourceAFIS](https://sourceafis.machinezoo.com/). Templates are stored in PostgreSQL. A small Bootstrap web page and a command-line client are included for using and testing the application. Custom model code and training are also included; see the [ML pipeline](ml_pipeline) folder.

## Features

- **Enroll:** stores a fingerprint for a subject ID. Enrollment is refused if the fingerprint already matches an enrolled subject, including the same one.
- **Verify (1:1):** checks whether a fingerprint matches a specific subject.
- **Identify (1:N):** search all enrolled fingerprints for the best match.
- **Web UI** served at `http://localhost:8080/`.
- **Command-line client** for scripted testing.

## Requirements

- Java 25
- PostgreSQL
- Internet access for Bootstrap CSS
- Maven

## Project layout

```
src/main/java/
  fingerprintapp/
    Application.java   Enrollment, verification, identification logic and SQL
    Server.java        Javalin HTTP server and routes
    Client.java        Command-line client
  commands/            Enroll, Verify, Identify
  results/             EnrollOK, EnrollFail, VerifyOK, IdentifyOK
src/main/resources/
  public/index.html    Web UI
  database-init.sql    Database schema
```

## Setup

### 1. Create the database

```
psql -U postgres -c "CREATE DATABASE fingerprints;"
psql -U postgres -d fingerprints -f src/main/resources/database-init.sql
```

> The schema script starts with `DROP TABLE IF EXISTS ... CASCADE`. Running it again deletes all enrolled data.

### 2. Configure `.env`

Configure global environment variables or create a `.env` file in the project root, adjusting values as necessary:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fingerprints
DB_USER=postgres
DB_PASSWORD=<your_password>
```

### 3. Run the server

Run `fingerprintapp.Server` from your IDE (or your build tool). The default port is 8080.

Open `http://localhost:8080/` in a browser.

## Using the web page

1. **Enroll:** enter a subject ID, choose a fingerprint image, and run enrollment.
2. **Verify:** enter the subject ID and choose a fingerprint image.
3. **Identify:** choose a fingerprint image only. The best match above the threshold is returned.

## Using the command-line client

Start the server first, then run `fingerprintapp.Client` with arguments:

```
enroll   <subjectID> <imagePath>
verify   <subjectID> <imagePath>
identify <imagePath>
```

To point the client at a different server:

```
java fingerprintapp.Client identify 101_3.tif
```

## HTTP API

All endpoints accept JSON. `image` is the Base64-encoded fingerprint image (TIFF, PNG, JPEG, etc.). Invalid Base64 or an unreadable image returns `400` with `{"error": "..."}`.

| Endpoint | Request | Response |
|---|---|---|
| `POST /api/enroll` | `{"subjectID": "101", "image": "..."}` | `{"subjectID": "101"}` on success, or `{"subjectID": "<matched>", "score": 198.6}` if refused as a duplicate |
| `POST /api/verify` | `{"subjectID": "101", "image": "..."}` | `{"verified": true, "score": 87.3}` |
| `POST /api/identify` | `{"image": "..."}` | `{"subjectID": "101", "score": 87.3}`, or `subjectID: null` if no subject meets the threshold |

## Sample testing with the FVC2000 DB1_B dataset

[FVC2000 DB1_B](https://github.com/robertvazan/fingerprint-datasets#fvc2000-db1-b) has 10 fingers with 8 impressions each. A sample, quick manual test:

1. Enroll `101_1.tif` as subject `101`. Expect success.
2. Enroll `101_1.tif` as `101` again. Expect "Already enrolled."
3. Enroll `101_1.tif` as `999`. Expect "Enrollment refused," naming `101`.
4. Verify subject `101` with `101_2.tif`. Expect a match with a high score.
5. Verify subject `101` with `102_2.tif`. Expect a rejection with a low score.
6. Identify with `101_3.tif`. Expect subject `101`.

## How matching works

- Images are converted to SourceAFIS templates on the server. Only templates are stored.
- A score of 40 or higher counts as a match. The threshold value is the default value according to SourceAFIS documentation.
- Identification compares the probe against every stored template, so the chance of a false match grows with the number of enrolled fingerprints.

## Limitations / TODO

- **Enrollment race:** the duplicate check runs outside the database transaction, so two identical enrollments submitted at the same instant could both succeed. The web page disables its button during a request to avoid accidental double submits.
- **One template per finger:** a print that matches an enrolled one is rejected as a duplicate, so a subject cannot add a second impression of the same finger. Different fingers can still be enrolled under the same subject ID.
- **Constant threshold:** verification and identification both use a match score of 40. This suits small galleries, but identification across thousands of templates needs a higher, separately tuned threshold.
- **Identification speed:** every stored template is re-parsed on each request, so response times grow linearly with the number of enrolled fingerprints.
- **Information exposure:** a refused enrollment reveals which subject ID the fingerprint matched. That is fine for local testing but would leak who is enrolled in a real system.
- **Request size:** Javalin limits request bodies to 1 MB by default, which fits the FVC images but may not fit larger scans from a USB reader.
- **Database connections:** `PGSimpleDataSource` opens a new connection per call. 
- **No security layer:** there is no authentication or HTTPS, so this is intended for local use only.
- **No automated tests:** testing is manual via the web page or the command-line client. Maven tests are included, but they are separate from the application *for now*.
- **Unencrypted storage:** fingerprint templates are stored as plain bytes in PostgreSQL. A leaked database or backup would expose them. 
- **Single-node design:** the application and database run as single instances with no replication, failover, or sharding. Backups are not automated. Identification scales by scanning all templates, so it is limited by one machine's CPU and memory.
