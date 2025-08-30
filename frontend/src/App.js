import React, { useState, useEffect } from 'react';
import './App.css';
import Header from './components/Header';
import NormalExamples from './components/NormalExamples';
import SidebarLeft from './components/SidebarLeft';
import SidebarRight from './components/SidebarRight';
import MainContent from './components/MainContent';
import { useCamera } from './hooks/useCamera';
import { useImageUpload } from './hooks/useImageUpload';
import { EYE_CONDITIONS } from './constants';

function App() {
  const [windowHeight, setWindowHeight] = useState(window.innerHeight);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const { 
    selectedFile, 
    previewUrl, 
    handleFileChange, 
    setSelectedFile, 
    setPreviewUrl 
  } = useImageUpload();
  
  const {
    cameraActive,
    videoRef,
    toggleCamera,
    captureImage,
  } = useCamera({ 
    onCapture: (file, url) => {
      setSelectedFile(file);
      setPreviewUrl(url);
      setPrediction(null);
      setError(null);
    }
  });

  // Handle window resize for responsive layout
  useEffect(() => {
    const handleResize = () => setWindowHeight(window.innerHeight);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleSubmit = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
        mode: 'cors'
      });
      
      if (!res.ok) throw new Error(`Status ${res.status}`);
      
      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      setError('Failed to get prediction.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ height: windowHeight }}>
      <Header />
      <NormalExamples eyeConditions={EYE_CONDITIONS} />
      
      <div className="content">
        <SidebarLeft eyeConditions={EYE_CONDITIONS} />
        
        <MainContent
          cameraActive={cameraActive}
          videoRef={videoRef}
          previewUrl={previewUrl}
          toggleCamera={toggleCamera}
          captureImage={captureImage}
          handleFileChange={handleFileChange}
          selectedFile={selectedFile}
          loading={loading}
          handleSubmit={handleSubmit}
          prediction={prediction}
          error={error}
        />
        
        <SidebarRight eyeConditions={EYE_CONDITIONS} />
      </div>
    </div>
  );
}

export default App;