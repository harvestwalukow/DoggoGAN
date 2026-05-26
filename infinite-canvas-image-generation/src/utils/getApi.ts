export const getApi = () => {
  if (import.meta.env.DEV) {
    return "http://localhost:3000";
  }
  return import.meta.env.VITE_API_URL || "";
};
