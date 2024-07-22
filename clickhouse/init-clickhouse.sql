CREATE DATABASE IF NOT EXISTS test_db;

CREATE USER IF NOT EXISTS test_user IDENTIFIED WITH plaintext_password BY 'test_password';

GRANT ALL ON test_db.* TO test_user;
