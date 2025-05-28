import React from "react";
import { Outlet } from "react-router-dom";
import { Header } from "../components/Header";
import { Footer } from "../components/Footer";
import "../styles/global.scss";

export const Layout = () => {
  return (
    <div className="app">
      <Header />
      <main className="gradient-background">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
};
