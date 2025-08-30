import React from 'react';
import FlipCard from './FlipCard';

const SidebarLeft = ({ eyeConditions }) => (
  <div className="sidebar left">
    <FlipCard 
      imageSrc="/1.jpg" 
      altText="Cataract" 
      className="side-image" 
      backContent={eyeConditions.cataract} 
      isCard={true}
    />
    
    <FlipCard 
      imageSrc="/2.jpg" 
      altText="Conjunctivitis" 
      className="side-image" 
      backContent={eyeConditions.conjunctivitis} 
      isCard={true}
    />
  </div>
);

export default SidebarLeft;