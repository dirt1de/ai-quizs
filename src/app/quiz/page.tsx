import Quiz from '../../components/Quiz';
import { ErrorBoundary } from '../../components/ErrorBoundary';

export default function QuizPage() {
  return (
    <main className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="py-20 text-center">
          <h1 className="text-5xl font-semibold tracking-tight text-gray-900 mb-6">
            Machine Learning & AI Quiz
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Test your knowledge of Machine Learning and Artificial Intelligence concepts.
            Get instant feedback and learn from detailed explanations with references.
          </p>
        </div>
        <ErrorBoundary>
          <Quiz />
        </ErrorBoundary>
      </div>
    </main>
  );
} 