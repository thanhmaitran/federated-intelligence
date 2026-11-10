import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview";
import Datasets from "./pages/Datasets";
import Clients from "./pages/Clients";
import Configure from "./pages/Configure";
import Monitor from "./pages/Monitor";
import Compare from "./pages/Compare";
import Privacy from "./pages/Privacy";
import Communication from "./pages/Communication";
import History from "./pages/History";

const NAV: [string, string][] = [
  ["/", "Overview"], ["/datasets", "Datasets"], ["/clients", "Clients"], ["/configure", "Experiment configuration"],
  ["/monitor", "Training monitor"], ["/compare", "Algorithm comparison"], ["/privacy", "Privacy analysis"],
  ["/communication", "Communication analysis"], ["/history", "Experiment history"],
];

export default function App() {
  return (
    <div className="layout">
      <nav className="side">
        <div className="brand">Federated Intelligence<small>Research dashboard</small></div>
        {NAV.map(([to, label]) => (
          <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>{label}</NavLink>
        ))}
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/clients" element={<Clients />} />
          <Route path="/configure" element={<Configure />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="/communication" element={<Communication />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}
