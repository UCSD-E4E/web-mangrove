import React from 'react';
import mapboxgl from 'mapbox-gl';
import styles from './index.module.scss';

mapboxgl.accessToken =
  'pk.eyJ1Ijoibm1laXN0ZXIiLCJhIjoiY2tjODZya3VnMHU0cjJ2bGpxanh0eW9idiJ9._DNCB5IcbFoGl7AIm0vVlA';

const Visualize: React.FC = () => {
  const mapContainer = React.useRef(null);
  React.useEffect(() => {
    const map = new mapboxgl.Map({
      container: mapContainer.current || '',
      style: 'mapbox://styles/mapbox/streets-v11',
      center: [-70.9, 42.35],
      zoom: 9,
    });
    map.on('load', () => {
      map.resize();
    });
    return () => map.remove();
  }, []);
  return (
    <div className={styles.mapContainer}>
      <div className={styles.map} ref={mapContainer} />
    </div>
  );
};

export default Visualize;
