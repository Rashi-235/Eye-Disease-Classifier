import { useState } from 'react';

export const useImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileChange = e => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  return {
    selectedFile,
    previewUrl,
    handleFileChange,
    setSelectedFile,
    setPreviewUrl
  };
};