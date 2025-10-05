# AI API Client

Simple client to communicate with the AI prediction API. Handles predictions, file uploads, and downloads.

## Quick Start

```javascript
import { ClientAPI } from './client_simple.js';

// Initialize client
const client = new ClientAPI('http://localhost:8000');

// Make a prediction
const features = [5.1, 3.5, 1.4, 0.2];
const result = await client.envoyerDonnees(features, 'user_123');
```

## Key Methods

### `envoyerDonnees(features, userId)`
Send data for prediction:
```javascript
const features = [5.1, 3.5, 1.4, 0.2]; // Must be 4 numbers
const result = await client.envoyerDonnees(features, 'user_123');
// Returns: { statut, prediction: { classe_predite, probabilites, confiance }, user_id }
```

### `uploadTxt(filePath, userId)`
Upload text file:
```javascript
const result = await client.uploadTxt('./myfile.txt', 'user_123');
// Returns: { status: 'uploaded', filename, user_id }
```

### `downloadTxt(filename, destPath)`
Download uploaded file:
```javascript
const savedPath = await client.downloadTxt('myfile.txt', './downloaded.txt');
```

## UI Integration Example

```javascript
// React/Next.js component example
function PredictionForm() {
  const [result, setResult] = useState(null);
  const client = new ClientAPI('http://localhost:8000');

  async function handleSubmit(e) {
    e.preventDefault();
    const features = [
      parseFloat(e.target.f1.value),
      parseFloat(e.target.f2.value),
      parseFloat(e.target.f3.value),
      parseFloat(e.target.f4.value)
    ];
    
    const prediction = await client.envoyerDonnees(features, 'user_123');
    setResult(prediction);
  }

  return (
    <form onSubmit={handleSubmit}>
      <input name="f1" type="number" step="0.1" required />
      <input name="f2" type="number" step="0.1" required />
      <input name="f3" type="number" step="0.1" required />
      <input name="f4" type="number" step="0.1" required />
      <button type="submit">Predict</button>
      
      {result && (
        <div>
          <p>Predicted class: {result.prediction.classe_predite}</p>
          <p>Confidence: {(result.prediction.confiance * 100).toFixed(2)}%</p>
        </div>
      )}
    </form>
  );
}
```

## Requirements
- Node.js environment
- Dependencies: `axios`, `form-data`
- Running AI API server (default: http://localhost:8000)

## Error Handling
- All methods return `null` on error
- Check server connection with `verifierConnexion()`
- Errors are logged to console with descriptive messages

## Security Notes
- Validate user input before sending to API
- Sanitize file paths for upload/download
- Consider adding authentication if needed