import axios from 'axios';
import fs from 'fs';
import path from 'path';
import FormData from 'form-data';

/**
 * Configuration de l'API
 */
const API_URL = 'http://localhost:8000';

/**
 * Classe client pour communiquer avec l'API
 */
class ClientAPI {
    constructor(baseURL) {
        this.client = axios.create({
            baseURL: baseURL,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    /**
     * Vérifier l'état de l'API
     */
    async verifierConnexion() {
        try {
            const response = await this.client.get('/health');
            console.log('✅ API connectée:', response.data);
            return response.data;
        } catch (error) {
            console.error('❌ Erreur de connexion:', error.message);
            return null;
        }
    }

    /**
     * Envoyer des données JSON à l'API et recevoir la réponse
     */
    async envoyerDonnees(features, userId = 'utilisateur_1') {
        try {
            // Préparer les données JSON
            const donneesJSON = {
                features: features,
                user_id: userId
            };

            console.log('\n📤 Envoi des données:', donneesJSON);

            // Envoyer la requête POST
            const response = await this.client.post('/predict', donneesJSON);

            console.log('✅ Réponse reçue avec succès\n');
            return response.data;

        } catch (error) {
            if (error.response) {
                console.error('❌ Erreur serveur:', error.response.data);
            } else if (error.request) {
                console.error('❌ Pas de réponse du serveur');
            } else {
                console.error('❌ Erreur:', error.message);
            }
            return null;
        }
    }

    /**
     * Afficher les résultats de manière formatée
     */
    afficherResultats(resultat) {
        if (!resultat) {
            console.log('❌ Aucun résultat à afficher\n');
            return;
        }

        console.log('═══════════════════════════════════════');
        console.log('           RÉSULTATS DE L\'IA          ');
        console.log('═══════════════════════════════════════');
        console.log(`Statut: ${resultat.statut}`);
        console.log(`User ID: ${resultat.user_id}`);
        console.log('───────────────────────────────────────');
        console.log(`Classe prédite: ${resultat.prediction.classe_predite}`);
        console.log(`Confiance: ${(resultat.prediction.confiance * 100).toFixed(2)}%`);
        console.log('\nProbabilités par classe:');
        resultat.prediction.probabilites.forEach((prob, idx) => {
            const barre = '█'.repeat(Math.floor(prob * 50));
            console.log(`  Classe ${idx}: ${(prob * 100).toFixed(2)}% ${barre}`);
        });
        console.log('═══════════════════════════════════════\n');
    }

    /**
     * Upload a .txt file to the API (/upload_txt)
     */
    async uploadTxt(filePath, userId = 'utilisateur_1') {
        try {
            if (!fs.existsSync(filePath)) {
                console.error('❌ File not found:', filePath);
                return null;
            }
            if (path.extname(filePath).toLowerCase() !== '.txt') {
                console.error('❌ Only .txt files are supported');
                return null;
            }

            const form = new FormData();
            form.append('file', fs.createReadStream(filePath));
            form.append('user_id', userId);

            const headers = form.getHeaders();

            const response = await this.client.post('/upload_txt', form, {
                headers: headers,
                maxContentLength: Infinity,
                maxBodyLength: Infinity
            });

            console.log('✅ Upload response:', response.data);
            return response.data;
        } catch (error) {
            console.error('❌ Upload failed:', error.message || error);
            return null;
        }
    }

    /**
     * Download a .txt file from the API (/download_txt/:filename) and save it to destPath
     */
    async downloadTxt(filename, destPath = null) {
        try {
            const safeName = encodeURIComponent(filename);
            const response = await this.client.get(`/download_txt/${safeName}`, { responseType: 'stream' });
            if (!destPath) destPath = `downloaded_${path.basename(filename)}`;
            const writer = fs.createWriteStream(destPath);
            response.data.pipe(writer);
            await new Promise((resolve, reject) => {
                writer.on('finish', resolve);
                writer.on('error', reject);
            });
            console.log('✅ File downloaded to', destPath);
            return destPath;
        } catch (error) {
            console.error('❌ Download failed:', error.message || error);
            return null;
        }
    }
}

/**
 * Fonction principale
 */
async function main() {
    console.log('🤖 Client API IA - Interface JavaScript\n');

    // Créer l'instance du client
    const client = new ClientAPI(API_URL);

    // Vérifier la connexion
    console.log('🔍 Vérification de la connexion à l\'API...');
    await client.verifierConnexion();

    // Exemple 1: Envoyer des données
    console.log('\n--- Test 1 ---');
    const features1 = [5.1, 3.5, 1.4, 0.2];
    const resultat1 = await client.envoyerDonnees(features1, 'user_test_1');
    client.afficherResultats(resultat1);

    // Exemple 2: Autres données
    console.log('--- Test 2 ---');
    const features2 = [6.7, 3.0, 5.2, 2.3];
    const resultat2 = await client.envoyerDonnees(features2, 'user_test_2');
    client.afficherResultats(resultat2);

    // Exemple 3: Données aléatoires
    console.log('--- Test 3 ---');
    const features3 = [
        Math.random() * 10,
        Math.random() * 10,
        Math.random() * 10,
        Math.random() * 10
    ].map(n => parseFloat(n.toFixed(2)));
    const resultat3 = await client.envoyerDonnees(features3, 'user_test_3');
    client.afficherResultats(resultat3);

    // -----------------------
    // Demo: upload & download a .txt file
    // -----------------------
    try {
        const demoFilename = 'demo_upload.txt';
        const demoContent = 'Ceci est un fichier de démonstration pour l\'upload via l\'API.';
        fs.writeFileSync(demoFilename, demoContent, { encoding: 'utf8' });
        console.log(`\n📁 Created demo file: ${demoFilename}`);

        console.log('\n📤 Uploading demo file to API...');
        const uploadResp = await client.uploadTxt(demoFilename, 'demo_user');
        console.log('Upload response:', uploadResp);

        if (uploadResp && uploadResp.status === 'uploaded') {
            const remoteName = uploadResp.filename;
            const dest = 'downloaded_' + remoteName;
            console.log(`\n📥 Downloading ${remoteName} from API to ${dest}...`);
            const saved = await client.downloadTxt(remoteName, dest);
            if (saved) console.log('Downloaded file saved at', saved);
            else console.log('Download failed');
        }
    } catch (err) {
        console.error('Demo upload/download failed:', err);
    }
}

// Exécuter le programme
main().catch(console.error);
