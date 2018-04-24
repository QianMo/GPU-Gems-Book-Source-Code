using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Storage;

namespace ReliefMapping360
{
    public class ReliefMapping : Microsoft.Xna.Framework.Game
    {
        GraphicsDeviceManager graphics;
        ContentManager content;

        GamePadState StatePadPrev;
        KeyboardState StateKeyboardPrev;

        Texture2D m_ColorTexture;
        Texture2D m_ReliefTexture;
        Texture2D m_FontTexture;

        Matrix m_World = Matrix.Identity;
        Matrix m_View = Matrix.Identity;
        Matrix m_Proj = Matrix.Identity;

        Model m_Model;

        Effect m_Effect;

        SpriteBatch m_SpriteBatch;

        Vector3 m_CameraPos = new Vector3(0, 0, -200);
        Vector3 m_LightPos = new Vector3(50, 200, -200);

        int m_Technique = 2;
        float m_TextureTile = 1;
        float m_RotSpeed = 2;
        float m_ZoomSpeed = 100;
        float m_DepthSpeed = 0.2f;
        float m_DepthScale = 0.15f;
        bool m_DepthBias = false;
        bool m_BorderClamp = false;

        public ReliefMapping()
        {
            graphics = new GraphicsDeviceManager(this);
            content = new ContentManager(Services);

            graphics.PreferredBackBufferWidth = 800;
            graphics.PreferredBackBufferHeight = 600;
        }
        protected override void Initialize()
        {
            base.Initialize();
        }

        protected override void LoadGraphicsContent(bool loadAllContent)
        {
            if (loadAllContent)
            {
                m_ColorTexture = content.Load<Texture2D>("Content/tile1_color");
                m_ReliefTexture = content.Load<Texture2D>("Content/tile1_relief");
                m_FontTexture = content.Load<Texture2D>("Content/fonts");

                m_Model = content.Load<Model>("Content/cube");

                m_Effect = content.Load<Effect>("Content/ReliefMapping");

                m_SpriteBatch = new SpriteBatch(graphics.GraphicsDevice);
            }
        }

        protected override void UnloadGraphicsContent(bool unloadAllContent)
        {
            if (unloadAllContent == true)
            {
                content.Unload();
            }
        }

        void UpdateInput(double ElapsedTime)
        {
            GamePadState StatePad = GamePad.GetState(PlayerIndex.One);
            KeyboardState StateKeyboard = Keyboard.GetState();

            if (StatePad.Buttons.Back == ButtonState.Pressed)
                Exit();

            float InputTime = (float)ElapsedTime;

            float RotIntensityX = StatePad.ThumbSticks.Left.X;
            float RotIntensityY = StatePad.ThumbSticks.Left.Y;

            if (StateKeyboard.IsKeyDown(Keys.Left))
                RotIntensityX -= 1.0f;
            if (StateKeyboard.IsKeyDown(Keys.Right))
                RotIntensityX += 1.0f;

            if (StateKeyboard.IsKeyDown(Keys.Up))
                RotIntensityY += 1.0f;
            if (StateKeyboard.IsKeyDown(Keys.Down))
                RotIntensityY -= 1.0f;

            Vector3 AxisX = new Vector3(m_View.M11, m_View.M21, m_View.M31);
            Vector3 AxisY = new Vector3(m_View.M12, m_View.M22, m_View.M32);

            float AngleX = m_RotSpeed * InputTime * RotIntensityX;
            float AngleY = m_RotSpeed * InputTime * RotIntensityY;

            Matrix RotX = Matrix.Identity;
            Matrix RotY = Matrix.Identity;

            if (Math.Abs(AngleX) > 0.0001f)
                RotX = Matrix.CreateFromAxisAngle(AxisY, AngleX);
            if (Math.Abs(AngleY) > 0.0001f)
                RotY = Matrix.CreateFromAxisAngle(AxisX, -AngleY);

            Matrix Rot = RotX * RotY;

            m_CameraPos = Vector3.Transform(m_CameraPos, Rot);



            RotIntensityX = StatePad.ThumbSticks.Right.X;
            RotIntensityY = StatePad.ThumbSticks.Right.Y;

            if (StateKeyboard.IsKeyDown(Keys.NumPad4))
                RotIntensityX -= 1.0f;
            if (StateKeyboard.IsKeyDown(Keys.NumPad6))
                RotIntensityX += 1.0f;

            if (StateKeyboard.IsKeyDown(Keys.NumPad8))
                RotIntensityY += 1.0f;
            if (StateKeyboard.IsKeyDown(Keys.NumPad2))
                RotIntensityY -= 1.0f;

            AngleX = m_RotSpeed * InputTime * RotIntensityX;
            AngleY = m_RotSpeed * InputTime * RotIntensityY;

            RotX = Matrix.Identity;
            RotY = Matrix.Identity;

            if (Math.Abs(AngleX) > 0.0001f)
                RotX = Matrix.CreateFromAxisAngle(AxisY, AngleX);
            if (Math.Abs(AngleY) > 0.0001f)
                RotY = Matrix.CreateFromAxisAngle(AxisX, -AngleY);

            Rot = RotX * RotY;

            m_LightPos = Vector3.Transform(m_LightPos, Rot);



            float dist = m_CameraPos.Length();

            float zoom = StatePad.Triggers.Left - StatePad.Triggers.Right;

            if (StateKeyboard.IsKeyDown(Keys.PageDown))
                zoom -= 1;
            if (StateKeyboard.IsKeyDown(Keys.PageUp))
                zoom += 1;

            dist += zoom * m_ZoomSpeed * InputTime;

            if (dist < 100)
                dist = 100;

            m_CameraPos.Normalize();
            m_CameraPos *= dist;




            if (StatePad.Buttons.A == ButtonState.Pressed ||
                StateKeyboard.IsKeyDown(Keys.End))
            {
                m_DepthScale -= InputTime * m_DepthSpeed;
                if (m_DepthScale < 0.01f)
                    m_DepthScale = 0.01f;
            }

            if (StatePad.Buttons.B == ButtonState.Pressed ||
                StateKeyboard.IsKeyDown(Keys.Home))
            {
                m_DepthScale += InputTime * m_DepthSpeed;
                if (m_DepthScale > 0.5f)
                    m_DepthScale = 0.5f;
            }

            if (StateKeyboard.IsKeyDown(Keys.Delete) == true &&
                StateKeyboardPrev.IsKeyDown(Keys.Delete) == false)
            {
                m_ColorTexture = content.Load<Texture2D>("Content/tile1_color");
                m_ReliefTexture = content.Load<Texture2D>("Content/tile1_relief");
            }

            if (StateKeyboard.IsKeyDown(Keys.Insert) == true &&
                StateKeyboardPrev.IsKeyDown(Keys.Insert) == false)
            {
                m_ColorTexture = content.Load<Texture2D>("Content/rockbump_color");
                m_ReliefTexture = content.Load<Texture2D>("Content/rockbump_relief");
            }

            if (StateKeyboard.IsKeyDown(Keys.Divide) == true &&
                StateKeyboardPrev.IsKeyDown(Keys.Divide) == false)
            {
                m_DepthBias = !m_DepthBias;
            }

            if (StateKeyboard.IsKeyDown(Keys.Multiply) == true &&
                StateKeyboardPrev.IsKeyDown(Keys.Multiply) == false)
            {
                m_BorderClamp = !m_BorderClamp;
            }
            if (StateKeyboard.IsKeyDown(Keys.Subtract) == true &&
                StateKeyboardPrev.IsKeyDown(Keys.Subtract) == false)
            {
                if (m_TextureTile > 1)
                    m_TextureTile -= 1;
            }

            if (StateKeyboard.IsKeyDown(Keys.Add) == true &&
                StateKeyboardPrev.IsKeyDown(Keys.Add) == false)
            {
                if (m_TextureTile < 10)
                    m_TextureTile += 1;
            }

            if (StateKeyboard.IsKeyDown(Keys.Enter) == true &&
                StateKeyboardPrev.IsKeyDown(Keys.Enter) == false)
            {
                m_Technique = (m_Technique + 1) % m_Effect.Techniques.Count;
            }

            // Process input only if connected and if the packet numbers differ.
            if (StatePad.IsConnected && StatePad.PacketNumber != StatePadPrev.PacketNumber)
            {
                if (StatePad.DPad.Up == ButtonState.Pressed &&
                    StatePadPrev.DPad.Up == ButtonState.Released)
                {
                    m_ColorTexture = content.Load<Texture2D>("Content/tile1_color");
                    m_ReliefTexture = content.Load<Texture2D>("Content/tile1_relief");
                }
                if (StatePad.DPad.Down == ButtonState.Pressed &&
                    StatePadPrev.DPad.Down == ButtonState.Released)
                {
                    m_ColorTexture = content.Load<Texture2D>("Content/rockbump_color");
                    m_ReliefTexture = content.Load<Texture2D>("Content/rockbump_relief");
                }

                if (StatePad.Buttons.X == ButtonState.Pressed &&
                    StatePadPrev.Buttons.X == ButtonState.Released)
                {
                    m_DepthBias = !m_DepthBias;
                }

                if (StatePad.Buttons.Y == ButtonState.Pressed &&
                    StatePadPrev.Buttons.Y == ButtonState.Released)
                {
                    m_BorderClamp = !m_BorderClamp;
                }

                if (StatePad.DPad.Left == ButtonState.Pressed &&
                    StatePadPrev.DPad.Left == ButtonState.Released)
                {
                    if (m_TextureTile > 1)
                        m_TextureTile -= 1;
                }

                if (StatePad.DPad.Right == ButtonState.Pressed &&
                    StatePadPrev.DPad.Right == ButtonState.Released)
                {
                    if (m_TextureTile < 10)
                        m_TextureTile += 1;
                }

                if (StatePad.Buttons.LeftShoulder == ButtonState.Pressed &&
                    StatePadPrev.Buttons.LeftShoulder == ButtonState.Released)
                {
                    if (m_Technique == 0)
                        m_Technique = m_Effect.Techniques.Count - 1;
                    else
                        m_Technique = m_Technique - 1;
                }

                if (StatePad.Buttons.RightShoulder == ButtonState.Pressed &&
                    StatePadPrev.Buttons.RightShoulder == ButtonState.Released)
                {
                    m_Technique = (m_Technique + 1) % m_Effect.Techniques.Count;
                }

                StatePadPrev = StatePad;
            }

            StateKeyboardPrev = StateKeyboard;
        }

        protected override void Update(GameTime gameTime)
        {
            UpdateInput(gameTime.ElapsedGameTime.TotalSeconds);

            base.Update(gameTime);
        }

        void DrawText(int x, int y, string digits, Vector4 color)
        {
            float xPosition = x;
            int Width = 12;

            for (int i = 0; i < digits.Length; i++)
            {
                if (digits[i] != ' ')
                {
                    int px = digits[i] % 16;
                    int py = digits[i] / 16;

                    m_SpriteBatch.Draw(
                        m_FontTexture,
                        new Vector2(xPosition, (float)y),
                        new Rectangle(px * 32, py * 32, 20, 20),
                        new Color(color));
                }

                xPosition += Width;
            }
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice gd = graphics.GraphicsDevice;
            gd.Clear(ClearOptions.DepthBuffer | ClearOptions.Target, Color.Gray, 1.0f, 0);

            gd.RenderState.CullMode = CullMode.CullCounterClockwiseFace;
            gd.RenderState.AlphaBlendEnable = true;
            gd.RenderState.SourceBlend = Blend.SourceAlpha;
            gd.RenderState.DestinationBlend = Blend.InverseSourceAlpha;

            Matrix[] Transforms = new Matrix[m_Model.Bones.Count];
            m_Model.CopyAbsoluteBoneTransformsTo(Transforms);

            m_View = Matrix.CreateLookAt(m_CameraPos, Vector3.Zero, Vector3.Up);
            m_Proj = Matrix.CreatePerspectiveFieldOfView(
                                MathHelper.ToRadians(60.0f),
                                (float)graphics.GraphicsDevice.Viewport.Width / (float)graphics.GraphicsDevice.Viewport.Height,
                                1.0f, 1000.0f);

            m_Effect.CurrentTechnique = m_Effect.Techniques[m_Technique];

            m_Effect.Begin();
            m_Effect.CurrentTechnique.Passes[0].Begin();

            m_Effect.Parameters["g_TextureTile"].SetValue(m_TextureTile);
            m_Effect.Parameters["g_ColorMap"].SetValue(m_ColorTexture);
            m_Effect.Parameters["g_ReliefMap"].SetValue(m_ReliefTexture);
            m_Effect.Parameters["g_BorderClamp"].SetValue(m_BorderClamp);
            m_Effect.Parameters["g_DepthBias"].SetValue(m_DepthBias);
            m_Effect.Parameters["g_DepthScale"].SetValue(m_DepthScale);

            foreach (ModelMesh Mesh in m_Model.Meshes)
            {
                m_World = Transforms[Mesh.ParentBone.Index];

                Matrix WorldViewProj = m_World * m_View * m_Proj;
                Matrix WorldInv = Matrix.Invert(m_World);

                Vector3 LocalLightPos = Vector3.Transform(m_LightPos, WorldInv);
                Vector3 LocalCameraPos = Vector3.Transform(m_CameraPos, WorldInv);

                m_Effect.Parameters["g_LightPos"].SetValue(LocalLightPos);
                m_Effect.Parameters["g_CameraPos"].SetValue(LocalCameraPos);
                m_Effect.Parameters["g_WorldViewProj"].SetValue(WorldViewProj);
                m_Effect.CommitChanges();

                foreach (ModelMeshPart MeshPart in Mesh.MeshParts)
                {
                    if (MeshPart.PrimitiveCount > 0)
                    {
                        gd.VertexDeclaration = MeshPart.VertexDeclaration;
                        gd.Vertices[0].SetSource(Mesh.VertexBuffer, MeshPart.StreamOffset, MeshPart.VertexStride);
                        gd.Indices = Mesh.IndexBuffer;
                        gd.DrawIndexedPrimitives(PrimitiveType.TriangleList, MeshPart.BaseVertex, 0, MeshPart.NumVertices, MeshPart.StartIndex, MeshPart.PrimitiveCount);
                    }
                }
            }

            m_Effect.CurrentTechnique.Passes[0].End();
            m_Effect.End();

            gd.RenderState.DepthBufferWriteEnable = false;
            gd.RenderState.DepthBufferEnable = false;

            string[] Techniques = new string[3] { "None", "Linear+Binary", "RelaxedCone+Binary" };
            m_SpriteBatch.Begin(SpriteBlendMode.AlphaBlend, SpriteSortMode.Deferred, SaveStateMode.None);
			DrawText(2, 2, "Relief Mapping", new Vector4(1, 0.4f, 0, 1));
            m_SpriteBatch.End();
            m_SpriteBatch.Begin(SpriteBlendMode.AlphaBlend, SpriteSortMode.Deferred, SaveStateMode.None);
            DrawText(0, 0, "Relief Mapping", Vector4.One);
            DrawText(0, 22, "Ray Intersect: " + Techniques[m_Technique], Vector4.One);
            DrawText(0, 44, "Depth Factor: " + m_DepthScale.ToString("N"), Vector4.One);
            DrawText(0, 66, "Depth Bias: " + m_DepthBias.ToString(), Vector4.One);
            DrawText(0, 88, "Border Clamp: " + m_BorderClamp.ToString(), Vector4.One);
            m_SpriteBatch.End();

            base.Draw(gameTime);
        }
    }
}