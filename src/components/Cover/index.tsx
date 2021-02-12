import { Button } from 'antd';
import React from 'react';
import styles from './index.module.scss';
import background from '../../assets/images/header.jpg';

interface CoverProps {
  height: string;
  isHome: boolean;
}

const Cover: React.FC<CoverProps> = (props) => {
  return (
    <div
      style={{ background: `url(${background})`, backgroundSize: 'cover', height: props.height }}
      className={styles.cover}
    >
      <h1 className={styles.header}>Mangrove Image Classification</h1>
      <Button type="primary" shape="round" size="large" hidden={!props.isHome}>
        Classify
      </Button>
    </div>
  );
};

export default Cover;
