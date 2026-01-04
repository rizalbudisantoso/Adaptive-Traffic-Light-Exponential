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
        die("❌ DB Connection failed: " . $conn->connect_error);
    }

    // MQTT setup
    $server = 'localhost';
    $port = 1883;
    $clientId = 'PHPSubscriber_' . uniqid();

    $mqtt = new MqttClient($server, $port, $clientId);
    $connectionSettings = (new ConnectionSettings)
        ->setKeepAliveInterval(60)
        ->setConnectTimeout(3);

    $mqtt->connect($connectionSettings);

    echo "✅ Connected to MQTT Broker (localhost:1883)\n";
    echo "✅ Connected to Database (atl_2025)\n";
    echo "📡 Listening for ESP32 requests...\n";
    echo "====================================================\n\n";

    // Subscribe to request topic
    $mqtt->subscribe('trafficlight/request', function ($topic, $message) use ($conn, $mqtt) {
        $timestamp = date('Y-m-d H:i:s');
        echo "[$timestamp] 📩 Received request: $message\n";

        // Query database
        $query = "SELECT * FROM algoritmadurasi ORDER BY id DESC LIMIT 1";
        $result = $conn->query($query);

        if (!$result) {
            echo "[$timestamp] ❌ Query Error: " . $conn->error . "\n\n";
            return;
        }

        if ($result->num_rows > 0) {
            $row = $result->fetch_assoc();
            
            // ✅ PERBAIKAN: Gunakan floatval() dan round() untuk 2 desimal
            $data = [
                'Barat' => round(floatval($row['Barat']), 2),
                'Selatan' => round(floatval($row['Selatan']), 2),
                'Timur' => round(floatval($row['Timur']), 2),
                'Utara' => round(floatval($row['Utara']), 2),
                'id' => (int)$row['id']
            ];

            $jsonData = json_encode($data);

            // Publish response
            $mqtt->publish('trafficlight/durations', $jsonData, 0);
            
            echo "[$timestamp] 📤 Sent: Barat={$data['Barat']}s, Selatan={$data['Selatan']}s, Timur={$data['Timur']}s, Utara={$data['Utara']}s\n";
            echo "====================================================\n\n";
        } else {
            echo "[$timestamp] ❌ No data in table 'algoritmadurasi'\n\n";
        }

    }, 0);

    // Keep running (loop forever)
    $mqtt->loop(true);
    $mqtt->disconnect();

} catch (Exception $e) {
    echo "❌ Error: " . $e->getMessage() . "\n";
}
?>
