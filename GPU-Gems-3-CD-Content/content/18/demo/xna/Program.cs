using System;

namespace ReliefMapping360
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        static void Main(string[] args)
        {
            using (ReliefMapping game = new ReliefMapping())
            {
                game.Run();
            }
        }
    }
}
