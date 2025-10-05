-- Show existing databases
SHOW DATABASES;

-- Create a new database if it doesn't exist
CREATE DATABASE IF NOT EXISTS mysql_db;

-- Show databases again to verify the new database
SHOW DATABASES;

-- Use the new database
USE mysql_db;

-- Create the products table if it doesn't exist
CREATE TABLE IF NOT EXISTS products (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

-- Show tables in the current database
SHOW TABLES;

-- Describe the products table
DESC products;

-- Show current content of the products table
SELECT * FROM products;

-- Insert data into the products table
INSERT INTO products (name) VALUES ('MacBook Air M3'), ('Samsung Galaxy Fold 3');

-- Show updated content of the products table
SELECT * FROM products;
