import { Link } from 'react-router-dom';

export default function Contact() {
  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-8">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-10 max-w-md text-center">
        <div className="inline-flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-lg">V</span>
          </div>
          <span className="font-bold text-xl text-slate-800">VCAI</span>
        </div>
        <h1 className="text-2xl font-bold text-slate-800">Talk to us about Enterprise</h1>
        <p className="mt-3 text-slate-500">
          For enterprise plans and custom support, reach out and our team will get back to you.
        </p>
        <a
          href="mailto:fitai.sub@gmail.com"
          className="inline-block mt-6 px-6 py-3 rounded-xl font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90"
        >
          Email us
        </a>
        <div className="mt-6">
          <Link to="/" className="text-sm text-blue-600 font-medium hover:underline">
            Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}
