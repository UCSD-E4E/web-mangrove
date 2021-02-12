import React from 'react';
import { Menu } from 'antd';
import styles from './index.module.scss';

interface NavbarProps {
  menuClickHandler: any;
}

const Navbar: React.FC<NavbarProps> = (props) => {
  return (
    <Menu mode="horizontal" className={styles.navbar} onClick={props.menuClickHandler}>
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
