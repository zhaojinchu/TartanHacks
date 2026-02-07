import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { Analytics } from "./pages/Analytics";
import { Dashboard } from "./pages/Dashboard";
import { Heatmaps } from "./pages/Heatmaps";
import { Predictions } from "./pages/Predictions";

const links = [
  { to: "/dashboard", label: "Monitoring" },
  { to: "/analytics", label: "Analytics" },
  { to: "/heatmaps", label: "Heatmaps" },
  { to: "/predictions", label: "Predictions" }
];

export default function App(): JSX.Element {
  return (
    <div className="app-shell">
      <header className="top-nav">
        <div className="brand">TartanHacks Bin Intel</div>
        <nav>
          {links.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/heatmaps" element={<Heatmaps />} />
        <Route path="/predictions" element={<Predictions />} />
      </Routes>
    </div>
  );
}
