import React from 'react';
import FlipCard from './FlipCard';

const NormalExamples = ({ eyeConditions }) => (
  <div className="normal-examples">
    <FlipCard 
      imageSrc="/4.jpg" 
      altText="Normal eye example 1" 
      className="normal-eye" 
      backContent={eyeConditions.normal} 
    />
    
    <FlipCard 
      imageSrc="/5.jpg" 
      altText="Normal eye example 2" 
      className="normal-eye" 
      backContent={eyeConditions.normal} 
    />
  </div>
);

export default NormalExamples;