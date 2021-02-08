import React from 'react';
import { Menu } from 'antd';
import styles from './index.module.scss';

const Navbar: React.FC = () => {
  return (
    <Menu mode="horizontal" className={styles.navbar}>
      <Menu.Item>Home</Menu.Item>
      <Menu.Item>Upload</Menu.Item>
      <Menu.Item>Classify</Menu.Item>
      <Menu.Item>Visualize</Menu.Item>
      <Menu.Item>Download</Menu.Item>
      <Menu.Item>About</Menu.Item>
    </Menu>
  );
};

export default Navbar;
