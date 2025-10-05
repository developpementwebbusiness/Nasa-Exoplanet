const https = require('https');
const http = require('http');

// Configuration des URLs
const BASE_URL = "http://localhost:8000";
const PREDICT_BATCH_URL = `${BASE_URL}/predict_batch`;
const TEST_URL = `${BASE_URL}/test_prediction`;

// Données d'exoplanètes
const valuesData = {
    data: [
        {
            'OrbitalPeriod': '11.521446064',
            'OPup': '1.9800000e-06',
            'OPdown': '-1.9800000e-06',
            'TransEpoch': '170.8396880',
            'TEup': '1.310000e-04',
            'TEdown': '-1.310000e-04',
            'Impact': '2.4830',
            'ImpactUp': '2.8510',
            'ImpactDown': '-0.6730',
            'TransitDur': '3.63990',
            'DurUp': '0.01140',
            'DurDown': '-0.01140',
            'TransitDepth': '1.7984e+04',
            'DepthUp': '3.190e+01',
            'DepthDown': '-3.190e+01',
            'PlanetRadius': '150.51',
            'RadiusUp': '3.976e+01',
            'RadiusDown': '-1.331e+01',
            'EquilibriumTemp': '753.0',
            'TempUp': null,
            'TempDown': null,
            'InsolationFlux': '75.88',
            'InsolationUp': '58.89',
            'InsolationDown': '-19.99',
            'TransitSNR': '622.10',
            'StellarEffTemp': '5795.00',
            'SteffUp': '155.00',
            'SteffDown': '-172.00',
            'StellarLogG': '4.554',
            'LogGUp': '0.033',
            'LogGDown': '-0.176',
            'StellarRadius': '0.8480',
            'SradUp': '0.2240',
            'SradDown': '-0.0750',
            'RA': '297.079930',
            'Dec': '47.597401',
            'KeplerMag': '15.472'
        },
        {
            'OrbitalPeriod': '19.403937760',
            'OPup': '2.0680000e-05',
            'OPdown': '-2.0680000e-05',
            'TransEpoch': '172.4842530',
            'TEup': '8.420000e-04',
            'TEdown': '-8.420000e-04',
            'Impact': '0.8040',
            'ImpactUp': '0.0070',
            'ImpactDown': '-0.0050',
            'TransitDur': '12.21550',
            'DurUp': '0.05980',
            'DurDown': '-0.05980',
            'TransitDepth': '8.9187e+03',
            'DepthUp': '5.330e+01',
            'DepthDown': '-5.330e+01',
            'PlanetRadius': '7.18',
            'RadiusUp': '7.600e-01',
            'RadiusDown': '-6.800e-01',
            'EquilibriumTemp': '523.0',
            'TempUp': null,
            'TempDown': null,
            'InsolationFlux': '17.69',
            'InsolationUp': '6.66',
            'InsolationDown': '-4.88',
            'TransitSNR': '214.70',
            'StellarEffTemp': '5043.00',
            'SteffUp': '151.00',
            'SteffDown': '-151.00',
            'StellarLogG': '4.591',
            'LogGUp': '0.072',
            'LogGDown': '-0.048',
            'StellarRadius': '0.6800',
            'SradUp': '0.0720',
            'SradDown': '-0.0650',
            'RA': '289.258210',
            'Dec': '47.635319',
            'KeplerMag': '15.487'
        },
        {
            'OrbitalPeriod': '19.221388942',
            'OPup': '1.1230000e-06',
            'OPdown': '-1.1230000e-06',
            'TransEpoch': '184.5521637',
            'TEup': '4.500000e-05',
            'TEdown': '-4.500000e-05',
            'Impact': '1.0650',
            'ImpactUp': '0.0310',
            'ImpactDown': '-0.0340',
            'TransitDur': '4.79843',
            'DurUp': '0.00235',
            'DurDown': '-0.00235',
            'TransitDepth': '7.4284e+04',
            'DepthUp': '2.190e+01',
            'DepthDown': '-2.190e+01',
            'PlanetRadius': '49.29',
            'RadiusUp': '1.603e+01',
            'RadiusDown': '-5.000e+00',
            'EquilibriumTemp': '698.0',
            'TempUp': null,
            'TempDown': null,
            'InsolationFlux': '55.97',
            'InsolationUp': '54.55',
            'InsolationDown': '-16.40',
            'TransitSNR': '2317.00',
            'StellarEffTemp': '6117.00',
            'SteffUp': '182.00',
            'SteffDown': '-200.00',
            'StellarLogG': '4.496',
            'LogGUp': '0.052',
            'LogGDown': '-0.208',
            'StellarRadius': '0.9470',
            'SradUp': '0.3080',
            'SradDown': '-0.0960',
            'RA': '295.814540',
            'Dec': '47.690350',
            'KeplerMag': '15.341'
        }
    ],
    user_id: "nodejs_test_client"
};

// Fonction utilitaire pour faire des requêtes HTTP avec callbacks
function makeRequest(url, method, data, callback) {
    const urlObj = new URL(url);
    const options = {
        hostname: urlObj.hostname,
        port: urlObj.port || (urlObj.protocol === 'https:' ? 443 : 80),
        path: urlObj.pathname + urlObj.search,
        method: method,
        headers: {
            'Content-Type': 'application/json',
        }
    };

    if (data) {
        const postData = JSON.stringify(data);
        options.headers['Content-Length'] = Buffer.byteLength(postData);
    }

    const clientModule = urlObj.protocol === 'https:' ? https : http;
    
    const req = clientModule.request(options, (res) => {
        let body = '';
        
        res.on('data', (chunk) => {
            body += chunk;
        });
        
        res.on('end', () => {
            try {
                const parsedBody = body ? JSON.parse(body) : {};
                callback(null, {
                    statusCode: res.statusCode,
                    data: parsedBody,
                    text: body
                });
            } catch (e) {
                callback(null, {
                    statusCode: res.statusCode,
                    data: null,
                    text: body
                });
            }
        });
    });
    
    req.on('error', (err) => {
        callback(err, null);
    });
    
    if (data) {
        req.write(JSON.stringify(data));
    }
    
    req.end();
}

// Test endpoint simple (GET) - Version callback
function testSimpleEndpoint(callback) {
    console.log("🧪 Test endpoint simple...");
    
    makeRequest(TEST_URL, 'GET', null, (err, response) => {
        if (err) {
            console.log(`   ❌ Erreur connexion: ${err.message}`);
            callback(false);
            return;
        }
        
        if (response.statusCode === 200) {
            console.log("   ✅ Test simple OK");
            console.log("   📊 Résultat:", response.data);
            callback(true);
        } else {
            console.log(`   ❌ Test simple failed: ${response.statusCode} - ${response.text}`);
            callback(false);
        }
    });
}

// Test batch prediction (POST) - Version callback
function testBatchPrediction(callback) {
    console.log("\n📦 Test batch prediction...");
    console.log(`   📊 ${valuesData.data.length} exoplanètes à traiter`);
    console.log("   🎯 Note: TempUp et TempDown sont ignorés (étaient null de toute façon)");
    
    makeRequest(PREDICT_BATCH_URL, 'POST', valuesData, (err, response) => {
        if (err) {
            console.log(`   ❌ Exception lors du test batch: ${err.message}`);
            callback(false);
            return;
        }
        
        if (response.statusCode === 200) {
            console.log("   ✅ Batch prediction réussie!");
            console.log(`   📈 Nombre de résultats: ${response.data.data?.length || 0}`);
            
            // Afficher chaque résultat
            if (response.data.data) {
                response.data.data.forEach((pred, i) => {
                    console.log(`      🌟 Exoplanète ${i + 1}:`);
                    console.log(`         - Name: ${pred.name || 'N/A'}`);
                    console.log(`         - Score: ${pred.score || 'N/A'}`);
                    console.log(`         - Label: ${pred.label || 'N/A'}`);
                });
            }
            
            callback(true);
        } else {
            console.log(`   ❌ Erreur ${response.statusCode}`);
            console.log("   📝 Détail:", response.text);
            callback(false);
        }
    });
}

// Test d'une seule exoplanète - Version callback
function testSingleExoplanet(exoplanetIndex, callback) {
    console.log(`\n🌟 Test d'une exoplanète unique (index ${exoplanetIndex})...`);
    
    const singleData = {
        data: [valuesData.data[exoplanetIndex]],
        user_id: `nodejs_single_test_${exoplanetIndex}`
    };
    
    makeRequest(PREDICT_BATCH_URL, 'POST', singleData, (err, response) => {
        if (err) {
            console.log(`   ❌ Exception: ${err.message}`);
            callback(false);
            return;
        }
        
        if (response.statusCode === 200) {
            console.log("   ✅ Prédiction unitaire réussie!");
            
            if (response.data.data && response.data.data[0]) {
                const pred = response.data.data[0];
                console.log("   🎯 Résultat:");
                console.log(`      - Name: ${pred.name}`);
                console.log(`      - Score: ${pred.score}`);
                console.log(`      - Label: ${pred.label}`);
            }
            
            callback(true);
        } else {
            console.log(`   ❌ Erreur ${response.statusCode}`);
            console.log("   📝 Détail:", response.text);
            callback(false);
        }
    });
}

// Afficher le mapping des features
function afficherMappingFeatures() {
    console.log("\n📋 Mapping des 35 features utilisés:");
    console.log("=".repeat(60));
    
    const featuresUtilises = [
        'OrbitalPeriod', 'OPup', 'OPdown', 'TransEpoch', 'TEup', 'TEdown',
        'Impact', 'ImpactUp', 'ImpactDown', 'TransitDur', 'DurUp', 'DurDown',
        'TransitDepth', 'DepthUp', 'DepthDown', 'PlanetRadius', 'RadiusUp', 'RadiusDown',
        'EquilibriumTemp', 'InsolationFlux', 'InsolationUp', 'InsolationDown',
        'TransitSNR', 'StellarEffTemp', 'SteffUp', 'SteffDown', 'StellarLogG', 'LogGUp', 'LogGDown',
        'StellarRadius', 'SradUp', 'SradDown', 'RA', 'Dec', 'KeplerMag'
    ];
    
    const featuresIgnores = ['TempUp', 'TempDown'];
    
    console.log("✅ Features utilisés (35):");
    featuresUtilises.forEach((feature, i) => {
        console.log(`   ${(i + 1).toString().padStart(2)}. ${feature}`);
    });
    
    console.log(`\n❌ Features ignorés (${featuresIgnores.length}):`);
    featuresIgnores.forEach(feature => {
        console.log(`   - ${feature} (était null dans les données)`);
    });
    
    console.log(`\n📊 Total: ${featuresUtilises.length} features → Compatible avec StandardScaler`);
}

// Test de performance avec callbacks
function testPerformance(callback) {
    console.log("\n⚡ Test de performance...");
    
    const iterations = 5;
    const startTime = Date.now();
    let successCount = 0;
    let completedCount = 0;
    
    for (let i = 0; i < iterations; i++) {
        makeRequest(PREDICT_BATCH_URL, 'POST', valuesData, (err, response) => {
            completedCount++;
            
            if (!err && response.statusCode === 200) {
                successCount++;
                process.stdout.write(`✓ `);
            } else {
                process.stdout.write(`✗ `);
            }
            
            if (completedCount === iterations) {
                const endTime = Date.now();
                const totalTime = endTime - startTime;
                
                console.log(`\n   📈 Résultats:`);
                console.log(`      - Succès: ${successCount}/${iterations}`);
                console.log(`      - Temps total: ${totalTime}ms`);
                console.log(`      - Temps moyen: ${Math.round(totalTime / iterations)}ms par requête`);
                
                callback(successCount === iterations);
            }
        });
    }
}

// Fonction principale avec callbacks en cascade
function main() {
    console.log("🚀 Test API avec 35 features (sans TempUp/TempDown) - Version Node.js (Callbacks)");
    console.log("=".repeat(80));
    
    // Afficher le mapping des features
    afficherMappingFeatures();
    
    // Test 1: Endpoint simple
    console.log(`\n🔍 Test de connexion à l'API...`);
    testSimpleEndpoint((simpleOk) => {
        if (!simpleOk) {
            console.log("\n❌ L'API n'est pas accessible. Vérifiez qu'elle est démarrée sur le port 8000.");
            process.exit(1);
            return;
        }
        
        // Test 2: Batch prediction
        testBatchPrediction((batchOk) => {
            // Test 3: Prédiction unitaire
            testSingleExoplanet(0, (singleOk) => {
                // Test 4: Performance
                testPerformance((perfOk) => {
                    // Résumé final
                    console.log("\n" + "=".repeat(80));
                    console.log("📊 Résumé des tests Node.js (Callbacks):");
                    console.log(`   🩺 API Health: ${simpleOk ? '✅' : '❌'}`);
                    console.log(`   📦 Batch Prediction: ${batchOk ? '✅' : '❌'}`);
                    console.log(`   🎯 Single Prediction: ${singleOk ? '✅' : '❌'}`);
                    console.log(`   ⚡ Performance Test: ${perfOk ? '✅' : '❌'}`);
                    
                    if (batchOk) {
                        console.log("\n🎉 Votre endpoint /predict_batch fonctionne parfaitement avec Node.js!");
                        console.log("   Style de programmation: Callbacks (compatible avec toutes versions Node.js)");
                        console.log("\n💡 Intégration suggérée:");
                        console.log("   - Express.js pour créer un serveur web");
                        console.log("   - Socket.io pour des prédictions en temps réel");
                        console.log("   - PM2 pour la production");
                    } else {
                        console.log("\n⚠️  Il y a encore des problèmes à résoudre.");
                        console.log("   Vérifiez que votre API FastAPI est bien démarrée sur http://localhost:8000");
                    }
                });
            });
        });
    });
}

// Export des fonctions pour usage en module
module.exports = {
    valuesData,
    testSimpleEndpoint,
    testBatchPrediction,
    testSingleExoplanet,
    afficherMappingFeatures,
    testPerformance,
    main
};

// Exécuter automatiquement si lancé directement
if (require.main === module) {
    main();
}