import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Switch, Route, withRouter } from 'react-router-dom';
import Home from './pages/Home';
import Navbar from './components/Navbar';
import reportWebVitals from './reportWebVitals';
import './styles/index.less';
import About from './pages/About';

const App = withRouter((props) => {
  const menuClickHandler = (event: any) => {
    const key = parseInt(event.key.split('_')[1] as string, 10);
    switch (key) {
      case 0:
        props.history.push('/');
        break;
      case 5:
        props.history.push('/about');
        break;
      default:
        break;
    }
  };
  return (
    <>
      <Navbar menuClickHandler={menuClickHandler} />
      <Switch>
        <Route exact path="/">
          <Home />
        </Route>
        <Route exact path="/about">
          <About />
        </Route>
      </Switch>
    </>
  );
});

ReactDOM.render(
  <React.StrictMode>
    <Router>
      <App />
    </Router>
  </React.StrictMode>,
  document.getElementById('root'),
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
