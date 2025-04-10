import Quiz from '../components/Quiz';

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-4xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">
          Machine Learning & AI Quiz
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Test your knowledge of Machine Learning and Artificial Intelligence concepts.
          Get instant feedback and learn from detailed explanations with references.
        </p>
        <Quiz />
      </div>
    </main>
  );
}
