<?php
require __DIR__ . '/vendor/autoload.php';

use PhpMqtt\Client\MqttClient;
use PhpMqtt\Client\ConnectionSettings;

// Database config
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "atl_2025";

try {
    // Connect to database
    $conn = new mysqli($servername, $username, $password, $dbname);
    if ($conn->connect_error) {
        die("❌ Connection failed: " . $conn->connect_error);
    }

    // MQTT setup
    $server = 'localhost';
    $port = 1883;
    $clientId = 'PHPPublisher_' . uniqid();

    $mqtt = new MqttClient($server, $port, $clientId);
    $connectionSettings = (new ConnectionSettings)
        ->setKeepAliveInterval(60)
        ->setConnectTimeout(3);

    // Connect to broker
    $mqtt->connect($connectionSettings);
    echo "✅ Connected to MQTT Broker\n";

    // Query database
    $query = "SELECT * FROM algoritmadurasi ORDER BY id DESC LIMIT 1";
    $result = $conn->query($query);

    if (!$result) {
        die("❌ Query Error: " . $conn->error . "\n");
    }

    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        
        // ✅ PERBAIKAN: Gunakan floatval() atau round() untuk 2 desimal
        $data = [
            'Barat' => round(floatval($row['Barat']), 2),
            'Selatan' => round(floatval($row['Selatan']), 2),
            'Timur' => round(floatval($row['Timur']), 2),
            'Utara' => round(floatval($row['Utara']), 2),
            'id' => (int)$row['id']
        ];

        $jsonData = json_encode($data);

        // Publish to MQTT topic
        $mqtt->publish('trafficlight/durations', $jsonData, 0);
        
        echo "📤 Published to MQTT:\n";
        echo "   Topic: trafficlight/durations\n";
        echo "   Data: " . $jsonData . "\n";
        echo "   Barat: {$data['Barat']}s, Selatan: {$data['Selatan']}s, Timur: {$data['Timur']}s, Utara: {$data['Utara']}s\n";
        echo "   Time: " . date('Y-m-d H:i:s') . "\n";
        echo "✅ Success!\n";
    } else {
        echo "❌ No data found in table 'algoritmadurasi'\n";
    }

    // Disconnect
    $mqtt->disconnect();

} catch (Exception $e) {
    echo "❌ Error: " . $e->getMessage() . "\n";
}

$conn->close();
?>
