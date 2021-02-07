import React from 'react';
import { Menu } from "antd";
import './index.scss';

const Navbar: React.FC = () => {
  return (
    <Menu mode="inline" className="navbar">
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