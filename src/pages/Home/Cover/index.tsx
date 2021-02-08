import { Button } from 'antd';
import React from 'react';
import styles from './index.module.scss';
import background from '../../../assets/images/header.jpg';

const Cover: React.FC = () => {
  return (
    <div
      style={{ background: `url(${background})`, backgroundSize: 'cover' }}
      className={styles.cover}
    >
      <h1 className={styles.header}>Mangrove Image Classification</h1>
      <Button type="primary" shape="round" size="large">
        Classify
      </Button>
    </div>
  );
};

export default Cover;
