// src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Set default base URL for all axios requests
axios.defaults.baseURL = 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('results');
  const [stats, setStats] = useState({ processed: 0, totalDetections: 0, uniqueClasses: 0, distribution: {} });
  const [modelInfo, setModelInfo] = useState({ health: null, version: null, classes: [] });
  const [apiResponse, setApiResponse] = useState(null);
  const [apiLoading, setApiLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  // New state for lightbox
  const [lightbox, setLightbox] = useState({
    isOpen: false,
    imgSrc: ''
  });

  useEffect(() => {
    // Fetch initial model info
    axios.get('/health/')
      .then(res => setModelInfo(info => ({ ...info, health: res.data })))
      .catch(() => {});
    axios.get('/version/')
      .then(res => setModelInfo(info => ({ ...info, version: res.data })))
      .catch(() => {});
    axios.get('/classes/')
      .then(res => setModelInfo(info => ({ ...info, classes: res.data.classes })))
      .catch(() => {});
  }, []);

  useEffect(() => {
    // Compute detection stats
    if (!results.length) {
      setStats({ processed: 0, totalDetections: 0, uniqueClasses: 0, distribution: {} });
      return;
    }
    const processed = results.length;
    let totalDetections = 0;
    const classCounts = {};
    results.forEach(r => {
      Object.values(r.results).forEach(m => {
        m.detections.forEach(d => {
          totalDetections++;
          classCounts[d.class] = (classCounts[d.class] || 0) + 1;
        });
      });
    });
    const uniqueClasses = Object.keys(classCounts).length;
    setStats({ processed, totalDetections, uniqueClasses, distribution: classCounts });
  }, [results]);

  // Close lightbox on escape key press
  useEffect(() => {
    const handleEscapeKey = (e) => {
      if (e.key === 'Escape' && lightbox.isOpen) {
        closeLightbox();
      }
    };
    
    document.addEventListener('keydown', handleEscapeKey);
    
    return () => {
      document.removeEventListener('keydown', handleEscapeKey);
    };
  }, [lightbox.isOpen]);

  const handleFileChange = e => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      // Create preview URL
      const fileReader = new FileReader();
      fileReader.onload = (e) => setPreviewUrl(e.target.result);
      fileReader.readAsDataURL(file);
    }
  };

  const handleDrop = e => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
      // Create preview URL
      const fileReader = new FileReader();
      fileReader.onload = (e) => setPreviewUrl(e.target.result);
      fileReader.readAsDataURL(file);
    }
  };

  const handleDragOver = e => {
    e.preventDefault();
  };

  const handleSubmit = async e => {
    e.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    setError('');
    setResults([]);
    setApiResponse(null);
    const formData = new FormData();
    formData.append('files', selectedFile);
    try {
      const res = await axios.post('/predict-all/', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setResults(res.data);
      setActiveTab('results'); // Switch to results tab after successful detection
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const callEndpoint = async endpoint => {
    setApiLoading(true);
    setApiResponse(null);
    try {
      const res = await axios.get(endpoint + '/');
      setApiResponse(res.data);
    } catch (err) {
      setApiResponse({ error: err.message });
    } finally {
      setApiLoading(false);
    }
  };

  const renderHighlightedJson = (json) => {
    if (!json) return 'No data';
    const stringified = JSON.stringify(json, null, 2);
    return stringified;
  };

  // Function to open lightbox
  const openLightbox = (imgSrc) => {
    setLightbox({
      isOpen: true,
      imgSrc
    });
    // Prevent body scrolling when lightbox is open
    document.body.style.overflow = 'hidden';
  };

  // Function to close lightbox
  const closeLightbox = () => {
    setLightbox({
      isOpen: false,
      imgSrc: ''
    });
    // Re-enable body scrolling
    document.body.style.overflow = 'auto';
  };

  return (
    <div className="App">
      <header className="hero">
        <div className="hero-content">
          <h1>X-Ray Detection</h1>
          <p>Upload baggage scans to detect prohibited items with AI precision</p>
          <div className="badges">
            <span><i className="fas fa-bolt"></i> Fast Processing</span>
            <span><i className="fas fa-bullseye"></i> High Accuracy</span>
            <span><i className="fas fa-search"></i> Multiple Models</span>
          </div>
        </div>
      </header>

      <main>
        <section className="upload-panel">
          <h2>Upload Baggage Scan Images</h2>
          <form onSubmit={handleSubmit} className="upload-form">
            <div 
              className={`dropzone ${selectedFile ? 'has-file' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <input type="file" onChange={handleFileChange} accept="image/*" id="file-input" />
              
              {previewUrl ? (
                <div className="preview-container">
                  <img 
                    src={previewUrl} 
                    alt="Preview" 
                    className="file-preview zoomable-image" 
                    onClick={() => openLightbox(previewUrl)}
                  />
                  <div className="file-name">{selectedFile?.name}</div>
                </div>
              ) : (
                <div className="dropzone-content">
                  <div className="upload-icon">
                    <i className="fas fa-cloud-upload-alt"></i>
                  </div>
                  <p>Drag & Drop your image here</p>
                  <span>or</span>
                  <label htmlFor="file-input" className="browse-button">Browse Files</label>
                </div>
              )}
            </div>
            
            <button 
              type="submit" 
              className={`detect-button ${loading ? 'loading' : ''}`} 
              disabled={loading || !selectedFile}
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <i className="fas fa-search"></i>
                  <span>Detect Objects</span>
                </>
              )}
            </button>
          </form>
          {error && <p className="error"><i className="fas fa-exclamation-circle"></i> {error}</p>}
        </section>

        <nav className="tabs">
          <button 
            className={activeTab === 'results' ? 'active' : ''} 
            onClick={() => setActiveTab('results')}
          >
            <i className="fas fa-camera"></i> Detection Results
          </button>
          <button 
            className={activeTab === 'stats' ? 'active' : ''} 
            onClick={() => setActiveTab('stats')}
          >
            <i className="fas fa-chart-bar"></i> Statistics
          </button>
          <button 
            className={activeTab === 'info' ? 'active' : ''} 
            onClick={() => setActiveTab('info')}
          >
            <i className="fas fa-info-circle"></i> Model Info
          </button>
          <button 
            className={activeTab === 'api' ? 'active' : ''} 
            onClick={() => setActiveTab('api')}
          >
            <i className="fas fa-code"></i> API Tester
          </button>
        </nav>

        <section className="content">
          {activeTab === 'results' && (
            <div className="results">
              {results.length ? (
                results.map((result, i) => (
                  <div key={i} className="result-card">
                    <h3>Detection Result {i+1}</h3>
                    <div className="result-images">
                      {Object.entries(result.results).map(([model, data]) => (
                        <div key={model} className="image-block">
                          <div className="model-label">{model}</div>
                          <div className="image-container">
                            <img 
                              src={`data:image/jpeg;base64,${data.image_base64}`} 
                              alt={model} 
                              className="zoomable-image"
                              onClick={() => openLightbox(`data:image/jpeg;base64,${data.image_base64}`)}
                            />
                            <div className="detection-count">
                              {data.detections.length} {data.detections.length === 1 ? 'detection' : 'detections'}
                            </div>
                          </div>
                          <div className="detections-list">
                            {data.detections.map((detection, index) => (
                              <div className="detection-item" key={index}>
                                <span className="detection-class">{detection.class}</span>
                                <span className="detection-confidence">
                                  {Math.round(detection.confidence * 100)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                <div className="placeholder">
                  <div className="placeholder-icon">
                    <i className="fas fa-camera"></i>
                  </div>
                  <p>No Results Yet</p>
                  <small>Upload an image and click "Detect Objects" to view results.</small>
                </div>
              )}
            </div>
          )}

          {activeTab === 'stats' && (
            <div className="stats-panel">
              <h2>Detection Statistics</h2>
              
              <div className="stats-overview">
                <div className="stat-card">
                  <div className="stat-icon">
                    <i className="fas fa-image"></i>
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">{stats.processed}</div>
                    <div className="stat-label">Processed Images</div>
                  </div>
                </div>
                
                <div className="stat-card">
                  <div className="stat-icon">
                    <i className="fas fa-search"></i>
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">{stats.totalDetections}</div>
                    <div className="stat-label">Total Detections</div>
                  </div>
                </div>
                
                <div className="stat-card">
                  <div className="stat-icon">
                    <i className="fas fa-tags"></i>
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">{stats.uniqueClasses}</div>
                    <div className="stat-label">Unique Classes</div>
                  </div>
                </div>
              </div>
              
              <h3>Detection Distribution</h3>
              <div className="distribution-chart">
                {Object.entries(stats.distribution).map(([cls, count]) => {
                  const percentage = (count / stats.totalDetections) * 100;
                  return (
                    <div className="stat-item bar" key={cls}>
                      <div className="bar-label">{cls}</div>
                      <div className="bar-container">
                        <div 
                          className="bar-fill" 
                          style={{ width: `${percentage}%` }}
                          data-percentage={`${percentage.toFixed(1)}%`}
                        />
                      </div>
                      <div className="bar-value">{count} ({percentage.toFixed(1)}%)</div>
                    </div>
                  );
                })}
                
                {Object.keys(stats.distribution).length === 0 && (
                  <div className="no-data">No detection data available</div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'info' && (
            <div className="info-panel">
              <h2>Model Information</h2>
              
              <div className="info-grid">
                <div className="info-item">
                  <div className="info-icon"><i className="fas fa-heartbeat"></i></div>
                  <div className="info-label">Health</div>
                  <div className={`info-value ${modelInfo.health?.status === 'ok' ? 'status-ok' : ''}`}>
                    {modelInfo.health?.status || 'Loading…'}
                  </div>
                </div>
                
                <div className="info-item">
                  <div className="info-icon"><i className="fas fa-code-branch"></i></div>
                  <div className="info-label">API Version</div>
                  <div className="info-value">{modelInfo.version?.api_version || '—'}</div>
                </div>
                
                <div className="info-item">
                  <div className="info-icon"><i className="fas fa-fire"></i></div>
                  <div className="info-label">Torch Version</div>
                  <div className="info-value">{modelInfo.version?.torch_version || '—'}</div>
                </div>
              </div>
              
              <div className="info-section">
                <h3><i className="fas fa-cube"></i> Available Models</h3>
                <div className="models-list">
                  {modelInfo.version?.models_available ? (
                    modelInfo.version.models_available.map(model => (
                      <div className="model-badge" key={model}>{model}</div>
                    ))
                  ) : (
                    <div className="loading-placeholder">Loading models...</div>
                  )}
                </div>
              </div>
              
              <div className="info-section">
                <h3><i className="fas fa-tags"></i> Detectable Classes</h3>
                <div className="class-list">
                  {modelInfo.classes.length > 0 ? (
                    modelInfo.classes.map(c => (
                      <span key={c} className="class-tag">{c}</span>
                    ))
                  ) : (
                    <div className="loading-placeholder">Loading classes...</div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'api' && (
            <div className="api-panel">
              <h2>Test API Endpoints</h2>
              <div className="api-buttons">
                <button
                  onClick={() => callEndpoint('/health')}
                  disabled={apiLoading}
                  className={`api-btn health ${apiLoading ? 'disabled' : ''}`}
                >
                  <i className="fas fa-heartbeat"></i> Health
                </button>
                
                <button
                  onClick={() => callEndpoint('/version')}
                  disabled={apiLoading}
                  className={`api-btn version ${apiLoading ? 'disabled' : ''}`}
                >
                  <i className="fas fa-code-branch"></i> Version
                </button>
                
                <button
                  onClick={() => callEndpoint('/classes')}
                  disabled={apiLoading}
                  className={`api-btn classes ${apiLoading ? 'disabled' : ''}`}
                >
                  <i className="fas fa-tags"></i> Classes
                </button>
                
                <button
                  onClick={() => callEndpoint('/predict-all')}
                  disabled={apiLoading}
                  className={`api-btn predict ${apiLoading ? 'disabled' : ''}`}
                >
                  <i className="fas fa-camera"></i> Predict All
                </button>
              </div>
              
              <div className="api-response-container">
                <div className="api-response-header">
                  <h3>Endpoint Response</h3>
                  {apiResponse && (
                    <div className="api-status">
                      Status: <span className="status-code">200 OK</span>
                    </div>
                  )}
                </div>
                
                <pre className="api-response">
                  {apiLoading ? (
                    <div className="api-loading">
                      <div className="spinner"></div>
                      <span>Fetching data...</span>
                    </div>
                  ) : (
                    renderHighlightedJson(apiResponse)
                  )}
                </pre>
              </div>
            </div>
          )}
        </section>
      </main>
      
      {/* Lightbox Component */}
      {lightbox.isOpen && (
        <div className="lightbox-overlay" onClick={closeLightbox}>
          <div className="lightbox-container" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={closeLightbox}>
              <i className="fas fa-times"></i>
            </button>
            <img src={lightbox.imgSrc} alt="Enlarged view" className="lightbox-image" />
          </div>
        </div>
      )}
      
      <footer className="app-footer">
        <p>X-Ray Detection System &copy; {new Date().getFullYear()}</p>
        <div className="footer-links">
          <a href="#about">About</a>
          <a href="#documentation">Documentation</a>
          <a href="#contact">Contact</a>
        </div>
      </footer>
    </div>
  );
}

export default App;