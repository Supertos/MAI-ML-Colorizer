// frontend/src/router/index.jsx
import { createBrowserRouter } from 'react-router-dom';
import { Layout } from '../app/Layout';
import { Home } from '../pages/Home';
import { Result } from '../pages/Result';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'result', element: <Result /> },
    ]
  }
]);
