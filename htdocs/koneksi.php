<?php
$host = "localhost";      // atau IP server database
$user = "root";           // ganti sesuai database
$pass = "";               // password database
$db   = "atl_2025";    // nama database

$conn = new mysqli($host, $user, $pass, $db);
if ($conn->connect_error) {
    die("Koneksi gagal: " . $conn->connect_error);
}
?>
